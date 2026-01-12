"""
Run Backtest
============
Portfolio backtesting simulation using ML predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import warnings

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import settings, print_welcome
from quant_alpha.data import DataLoader
from quant_alpha.features.registry import FactorRegistry
from quant_alpha.models import WalkForwardTrainer

warnings.filterwarnings('ignore')


class SimpleBacktester:
    """
    Simple portfolio backtesting engine.
    
    Features:
        - Long-only portfolio
        - Monthly rebalancing
        - Transaction costs
        - Performance metrics
    """
    
    def __init__(self):
        """Initialize backtester."""
        self.config = settings.backtest
        self.results = []
        self.portfolio_history = []
        
    def run_backtest(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run portfolio backtest.
        
        Args:
            predictions_df: DataFrame with predictions and returns
            
        Returns:
            Portfolio performance DataFrame
        """
        print("\n" + "="*70)
        print("📈 PORTFOLIO BACKTESTING")
        print("="*70)
        
        # Validate input
        required_cols = ['date', 'ticker', 'predictions', 'forward_return']
        missing_cols = set(required_cols) - set(predictions_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Initialize portfolio
        portfolio_value = self.config.initial_capital
        positions = {}
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(predictions_df)
        print(f"📅 Rebalancing dates: {len(rebalance_dates)}")
        
        portfolio_history = []
        
        for i, rebal_date in enumerate(rebalance_dates):
            print(f"   📊 Rebalancing {i+1}/{len(rebalance_dates)}: {pd.to_datetime(rebal_date).date()}")
            
            # Get predictions for this date
            date_data = predictions_df[predictions_df['date'] == rebal_date].copy()
            
            if len(date_data) == 0:
                continue
            
            # Select top stocks
            top_stocks = self._select_stocks(date_data)
            
            # Calculate new positions
            new_positions = self._calculate_positions(top_stocks, portfolio_value)
            
            # Calculate transaction costs
            transaction_cost = self._calculate_transaction_costs(positions, new_positions, portfolio_value)
            
            # Update portfolio
            portfolio_value -= transaction_cost
            positions = new_positions
            
            # Calculate returns until next rebalance
            if i < len(rebalance_dates) - 1:
                next_date = rebalance_dates[i + 1]
                period_return = self._calculate_period_return(positions, predictions_df, rebal_date, next_date)
                portfolio_value *= (1 + period_return)
            
            # Record portfolio state
            portfolio_history.append({
                'date': rebal_date,
                'portfolio_value': portfolio_value,
                'n_positions': len(positions),
                'transaction_cost': transaction_cost,
                'top_stocks': list(top_stocks['ticker'].values) if len(top_stocks) > 0 else []
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(portfolio_history)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(results_df)
        
        # Print summary
        self._print_backtest_summary(results_df, metrics)
        
        return results_df, metrics
    
    def _get_rebalance_dates(self, df: pd.DataFrame) -> list:
        """Get monthly rebalancing dates."""
        df['date'] = pd.to_datetime(df['date'])
        
        # Get month-end dates
        monthly_dates = df.groupby([df['date'].dt.year, df['date'].dt.month])['date'].max()
        
        return sorted(monthly_dates.values)
    
    def _select_stocks(self, date_data: pd.DataFrame) -> pd.DataFrame:
        """Select top N stocks based on predictions."""
        # Remove NaN predictions
        clean_data = date_data.dropna(subset=['predictions'])
        
        if len(clean_data) == 0:
            return pd.DataFrame()
        
        # Sort by predictions (descending)
        sorted_data = clean_data.sort_values('predictions', ascending=False)
        
        # Select top N
        top_n = min(self.config.top_n_long, len(sorted_data))
        
        return sorted_data.head(top_n)
    
    def _calculate_positions(self, top_stocks: pd.DataFrame, portfolio_value: float) -> dict:
        """Calculate position sizes."""
        if len(top_stocks) == 0:
            return {}
        
        # Equal weight positions
        weight_per_stock = 1.0 / len(top_stocks)
        max_weight = self.config.max_position_pct if hasattr(self.config, 'max_position_pct') else 0.15
        
        # Ensure no position exceeds max weight
        actual_weight = min(weight_per_stock, max_weight)
        
        positions = {}
        for _, row in top_stocks.iterrows():
            ticker = row['ticker']
            position_value = portfolio_value * actual_weight
            positions[ticker] = position_value
        
        return positions
    
    def _calculate_transaction_costs(self, old_positions: dict, new_positions: dict, portfolio_value: float) -> float:
        """Calculate transaction costs."""
        total_turnover = 0.0
        
        # Calculate turnover
        all_tickers = set(old_positions.keys()) | set(new_positions.keys())
        
        for ticker in all_tickers:
            old_weight = old_positions.get(ticker, 0) / portfolio_value if portfolio_value > 0 else 0
            new_weight = new_positions.get(ticker, 0) / portfolio_value if portfolio_value > 0 else 0
            
            turnover = abs(new_weight - old_weight)
            total_turnover += turnover
        
        # Apply transaction cost
        transaction_cost = total_turnover * portfolio_value * self.config.total_cost_pct
        
        return transaction_cost
    
    def _calculate_period_return(self, positions: dict, df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
        """Calculate portfolio return for a period."""
        if not positions:
            return 0.0
        
        total_return = 0.0
        total_weight = 0.0
        
        for ticker, position_value in positions.items():
            # Get stock data for this period
            stock_data = df[
                (df['ticker'] == ticker) & 
                (df['date'] >= start_date) & 
                (df['date'] < end_date)
            ].copy()
            
            if len(stock_data) == 0:
                continue
            
            # Use forward return if available
            if 'forward_return' in stock_data.columns:
                stock_return = stock_data['forward_return'].iloc[0]
                if not pd.isna(stock_return):
                    weight = position_value / sum(positions.values())
                    total_return += weight * stock_return
                    total_weight += weight
        
        return total_return / total_weight if total_weight > 0 else 0.0
    
    def _calculate_metrics(self, results_df: pd.DataFrame) -> dict:
        """Calculate portfolio performance metrics."""
        if len(results_df) < 2:
            return {}
        
        # Calculate returns
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        
        # Remove first NaN
        returns = results_df['returns'].dropna()
        
        if len(returns) == 0:
            return {}
        
        # Calculate metrics
        total_return = (results_df['portfolio_value'].iloc[-1] / results_df['portfolio_value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (12 / len(returns)) - 1  # Monthly data
        volatility = returns.std() * np.sqrt(12)  # Annualized
        sharpe_ratio = (annualized_return - settings.backtest.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_periods': len(returns),
            'avg_monthly_return': returns.mean(),
            'final_value': results_df['portfolio_value'].iloc[-1]
        }
        
        return metrics
    
    def _print_backtest_summary(self, results_df: pd.DataFrame, metrics: dict):
        """Print backtest summary."""
        print(f"\n📊 BACKTEST SUMMARY")
        print("="*50)
        
        if not metrics:
            print("❌ No metrics calculated!")
            return
        
        print(f"📅 Period: {results_df['date'].min().date()} → {results_df['date'].max().date()}")
        print(f"💰 Initial Capital: ${self.config.initial_capital:,.0f}")
        print(f"💰 Final Value: ${metrics['final_value']:,.0f}")
        print(f"📈 Total Return: {metrics['total_return']:.2%}")
        print(f"📈 Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"📊 Volatility: {metrics['volatility']:.2%}")
        print(f"⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"🎯 Win Rate: {metrics['win_rate']:.2%}")
        print(f"📊 Avg Monthly Return: {metrics['avg_monthly_return']:.2%}")
        print("="*50)


def main():
    """Run backtesting pipeline."""
    
    start_time = time.time()
    
    print_welcome()
    print("\n🎯 BACKTESTING PIPELINE")
    print("="*50)
    
    try:
        # Check if we have validation results
        results_path = settings.results_dir / "validation_results.csv"
        features_path = settings.data.processed_dir / "features_dataset.pkl"
        
        if not results_path.exists() or not features_path.exists():
            print("❌ Missing validation results! Run research pipeline first:")
            print("   python scripts/run_research.py")
            return False
        
        # Load data
        print("📊 Loading validation results...")
        validation_results = pd.read_csv(results_path)
        features_df = pd.read_pickle(features_path)
        
        print(f"✅ Loaded validation results: {len(validation_results)} folds")
        print(f"✅ Loaded features: {features_df.shape}")
        
        # Create predictions dataset (simplified approach)
        print("🔮 Generating predictions...")
        
        # For demo, use a simple approach - in practice, use out-of-sample predictions
        # This is a simplified version - real implementation would use proper walk-forward predictions
        
        # Get a subset of data for backtesting
        backtest_data = features_df.copy()
        
        # Simple prediction: use momentum as proxy (for demo)
        if 'mom_21d' in backtest_data.columns:
            backtest_data['predictions'] = backtest_data['mom_21d']
        else:
            # Fallback: random predictions
            np.random.seed(42)
            backtest_data['predictions'] = np.random.randn(len(backtest_data))
        
        # Run backtest
        backtester = SimpleBacktester()
        portfolio_df, metrics = backtester.run_backtest(backtest_data)
        
        # Save results
        backtest_results_path = settings.results_dir / "backtest_results.csv"
        portfolio_df.to_csv(backtest_results_path, index=False)
        
        # Save metrics
        metrics_path = settings.results_dir / "backtest_metrics.json"
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"\n✅ Backtest completed!")
        print(f"📊 Results saved: {backtest_results_path.name}")
        print(f"📈 Metrics saved: {metrics_path.name}")
        
        total_time = time.time() - start_time
        print(f"⏱️  Execution time: {total_time/60:.1f} minutes")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Portfolio Backtesting')
    parser.add_argument('--start-date', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, help='Initial capital', default=1000000)
    
    args = parser.parse_args()
    
    # Override config if provided
    if args.capital:
        settings.backtest.initial_capital = args.capital
    
    success = main()
    if success:
        print("\n🎯 Backtesting completed successfully!")
    else:
        print("\n💥 Backtesting failed!")
        sys.exit(1)