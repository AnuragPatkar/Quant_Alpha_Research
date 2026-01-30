"""
Backtesting Engine
==================
Simulates trading strategy with realistic costs.
Tests how the model would have performed in real trading.

Classes:
    - BacktestResult: Container for backtest results
    - Backtester: Main backtesting engine
    - BacktestConfig: Configuration dataclass

Features:
    - Long/Short strategies
    - Transaction costs
    - Multiple rebalancing frequencies
    - Position sizing
    - Slippage modeling
    - Detailed trade logging

Author: Senior Quant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
import logging
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from config.settings import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    logger.warning("Settings not available, using defaults")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Strategy
    top_n_long: int = 10
    top_n_short: int = 0  # 0 = long only
    
    # Costs
    transaction_cost_bps: float = 30.0  # 30 bps = 0.30%
    slippage_bps: float = 10.0  # 10 bps = 0.10%
    
    # Rebalancing
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly'
    
    # Portfolio
    initial_capital: float = 1_000_000.0
    max_position_size: float = 0.20  # Max 20% per position
    
    # Risk
    stop_loss_pct: Optional[float] = None  # e.g., 0.10 = 10% stop loss
    take_profit_pct: Optional[float] = None  # e.g., 0.20 = 20% take profit
    
    @property
    def total_cost_pct(self) -> float:
        """Total transaction cost as percentage."""
        return (self.transaction_cost_bps + self.slippage_bps) / 10000
    
    @property
    def total_cost_bps(self) -> float:
        """Total transaction cost in basis points."""
        return self.transaction_cost_bps + self.slippage_bps


@dataclass
class BacktestResult:
    """
    Container for backtest results.
    
    Attributes:
        returns: Period returns series
        cumulative: Cumulative returns series
        equity_curve: Portfolio value over time
        metrics: Performance metrics dictionary
        positions: Position history DataFrame
        trades: Trade log DataFrame
        config: Backtest configuration used
    """
    returns: pd.Series
    cumulative: pd.Series
    equity_curve: pd.Series = None
    metrics: Dict[str, float] = field(default_factory=dict)
    positions: pd.DataFrame = None
    trades: pd.DataFrame = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Create equity curve if not provided."""
        if self.equity_curve is None and self.cumulative is not None:
            initial_capital = self.config.get('initial_capital', 1_000_000)
            self.equity_curve = self.cumulative * initial_capital
    
    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "\n" + "="*60,
            "  BACKTEST RESULTS SUMMARY",
            "="*60,
            f"  Total Return:      {self.metrics.get('total_return', 0):>10.2%}",
            f"  Annual Return:     {self.metrics.get('annual_return', 0):>10.2%}",
            f"  Sharpe Ratio:      {self.metrics.get('sharpe_ratio', 0):>10.2f}",
            f"  Max Drawdown:      {self.metrics.get('max_drawdown', 0):>10.2%}",
            f"  Win Rate:          {self.metrics.get('win_rate', 0):>10.2%}",
            f"  Periods:           {self.metrics.get('n_periods', 0):>10}",
            "="*60
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'returns': self.returns,
            'cumulative': self.cumulative,
            'equity_curve': self.equity_curve,
            'metrics': self.metrics,
            'positions': self.positions,
            'trades': self.trades,
            'config': self.config
        }


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """
    Portfolio backtester with realistic simulation.
    
    Features:
        - Long/Short strategies
        - Transaction costs and slippage
        - Multiple rebalancing frequencies
        - Position tracking and trade logging
        - Performance metrics calculation
    
    Example:
        >>> backtester = Backtester()
        >>> result = backtester.run(predictions_df)
        >>> print(result.summary())
        
        # With custom config
        >>> config = BacktestConfig(top_n_long=20, transaction_cost_bps=20)
        >>> backtester = Backtester(config=config)
        >>> result = backtester.run(predictions_df)
    """
    
    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        verbose: bool = True
    ):
        """
        Initialize backtester.
        
        Args:
            config: BacktestConfig object (uses settings if None)
            verbose: Print progress and results
        """
        self.verbose = verbose
        
        # Load config
        if config is not None:
            self.config = config
        elif SETTINGS_AVAILABLE:
            self.config = BacktestConfig(
                top_n_long=settings.backtest.top_n_long,
                top_n_short=getattr(settings.backtest, 'top_n_short', 0),
                transaction_cost_bps=getattr(settings.backtest, 'transaction_cost_bps', 
                                            settings.backtest.total_cost_bps / 2),
                slippage_bps=getattr(settings.backtest, 'slippage_bps', 
                                    settings.backtest.total_cost_bps / 2),
                rebalance_frequency=settings.backtest.rebalance_frequency,
                initial_capital=getattr(settings.backtest, 'initial_capital', 1_000_000)
            )
        else:
            self.config = BacktestConfig()
        
        # Internal state
        self._trades_log: List[Dict] = []
        self._positions_log: List[Dict] = []
    
    def run(
        self,
        predictions: pd.DataFrame,
        benchmark: Optional[pd.Series] = None
    ) -> Optional[BacktestResult]:
        """
        Run backtest on predictions.
        
        Args:
            predictions: DataFrame with columns:
                - date: Trade date
                - ticker: Stock symbol
                - prediction: Model prediction/score
                - forward_return: Actual forward return
            benchmark: Optional benchmark returns series
            
        Returns:
            BacktestResult object or None if backtest fails
        """
        # Validate input
        if not self._validate_predictions(predictions):
            return None
        
        # Print header
        if self.verbose:
            self._print_header()
        
        # Reset logs
        self._trades_log = []
        self._positions_log = []
        
        # Get rebalance dates
        rebal_dates = self._get_rebalance_dates(predictions)
        
        if len(rebal_dates) < 2:
            logger.error("Insufficient rebalance dates")
            if self.verbose:
                print("\n   ⚠️ Insufficient data for backtesting!")
            return None
        
        # Run simulation
        returns_list = []
        previous_positions = set()
        
        for i in range(len(rebal_dates) - 1):
            rebal_date = rebal_dates[i]
            next_date = rebal_dates[i + 1]
            
            # Get predictions for rebalance date
            day_preds = predictions[predictions['date'] == rebal_date].copy()
            
            if len(day_preds) < self.config.top_n_long:
                continue
            
            # Select positions
            period_result = self._execute_period(
                day_preds, 
                rebal_date, 
                next_date,
                previous_positions
            )
            
            if period_result is not None:
                returns_list.append(period_result)
                previous_positions = period_result['positions']
        
        if not returns_list:
            if self.verbose:
                print("\n   ⚠️ No trades executed!")
            return None
        
        # Create result
        result = self._create_result(returns_list, predictions, benchmark)
        
        # Print results
        if self.verbose:
            self._print_results(result.metrics)
        
        return result
    
    def _validate_predictions(self, predictions: pd.DataFrame) -> bool:
        """Validate predictions DataFrame."""
        required_cols = ['date', 'ticker', 'prediction', 'forward_return']
        
        if predictions is None or len(predictions) == 0:
            logger.error("Predictions DataFrame is empty")
            return False
        
        missing = [col for col in required_cols if col not in predictions.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
            return False
        
        # Check for valid data
        if predictions['prediction'].isna().all():
            logger.error("All predictions are NaN")
            return False
        
        return True
    
    def _get_rebalance_dates(self, predictions: pd.DataFrame) -> List:
        """Get rebalance dates based on frequency."""
        dates = pd.to_datetime(predictions['date'].unique())
        dates = sorted(dates)
        
        if self.config.rebalance_frequency == 'daily':
            return dates
        
        elif self.config.rebalance_frequency == 'weekly':
            # First trading day of each week
            date_series = pd.Series(dates)
            weekly_dates = date_series.groupby(
                date_series.dt.to_period('W')
            ).first().values
            return list(weekly_dates)
        
        elif self.config.rebalance_frequency == 'monthly':
            # First trading day of each month
            date_series = pd.Series(dates)
            monthly_dates = date_series.groupby(
                date_series.dt.to_period('M')
            ).first().values
            return list(monthly_dates)
        
        else:
            logger.warning(f"Unknown frequency: {self.config.rebalance_frequency}, using monthly")
            date_series = pd.Series(dates)
            return list(date_series.groupby(date_series.dt.to_period('M')).first().values)
    
    def _execute_period(
        self,
        day_preds: pd.DataFrame,
        rebal_date: pd.Timestamp,
        next_date: pd.Timestamp,
        previous_positions: set
    ) -> Optional[Dict]:
        """
        Execute single rebalance period.
        
        Returns dict with 'date', 'return', 'positions', 'turnover'
        """
        # Sort by prediction
        day_preds = day_preds.sort_values('prediction', ascending=False)
        
        # Select long positions
        long_stocks = day_preds.head(self.config.top_n_long)
        
        # Select short positions (if enabled)
        if self.config.top_n_short > 0:
            short_stocks = day_preds.tail(self.config.top_n_short)
        else:
            short_stocks = pd.DataFrame()
        
        # Calculate positions
        current_positions = set(long_stocks['ticker'].tolist())
        if len(short_stocks) > 0:
            short_tickers = set(short_stocks['ticker'].tolist())
        else:
            short_tickers = set()
        
        # Calculate turnover
        all_positions = current_positions.union(short_tickers)
        if previous_positions:
            turnover = len(previous_positions.symmetric_difference(all_positions)) / \
                      max(len(previous_positions.union(all_positions)), 1)
        else:
            turnover = 1.0  # First period = 100% turnover
        
        # Calculate return
        long_return = long_stocks['forward_return'].mean() if len(long_stocks) > 0 else 0.0
        
        if len(short_stocks) > 0:
            # Short positions: profit from negative returns
            short_return = -short_stocks['forward_return'].mean()
            
            # Weight based on number of positions
            total_positions = len(long_stocks) + len(short_stocks)
            long_weight = len(long_stocks) / total_positions
            short_weight = len(short_stocks) / total_positions
            
            period_return = long_weight * long_return + short_weight * short_return
        else:
            period_return = long_return
        
        # Apply transaction cost (proportional to turnover)
        transaction_cost = turnover * self.config.total_cost_pct
        period_return -= transaction_cost
        
        # Log trades
        self._log_trades(rebal_date, long_stocks, short_stocks, previous_positions)
        
        # Log positions
        self._positions_log.append({
            'date': rebal_date,
            'stocks': list(current_positions),
            'n_long': len(long_stocks),
            'n_short': len(short_stocks),
            'turnover': turnover
        })
        
        return {
            'date': next_date,
            'return': period_return,
            'positions': all_positions,
            'turnover': turnover,
            'n_positions': len(all_positions),
            'long_return': long_return,
            'short_return': -short_stocks['forward_return'].mean() if len(short_stocks) > 0 else 0.0
        }
    
    def _log_trades(
        self,
        date: pd.Timestamp,
        long_stocks: pd.DataFrame,
        short_stocks: pd.DataFrame,
        previous_positions: set
    ):
        """Log individual trades."""
        current_long = set(long_stocks['ticker'].tolist())
        current_short = set(short_stocks['ticker'].tolist()) if len(short_stocks) > 0 else set()
        
        # New longs
        for ticker in current_long - previous_positions:
            pred_row = long_stocks[long_stocks['ticker'] == ticker].iloc[0]
            self._trades_log.append({
                'date': date,
                'ticker': ticker,
                'side': 'BUY',
                'type': 'LONG',
                'prediction': pred_row['prediction'],
                'forward_return': pred_row['forward_return']
            })
        
        # Closed longs
        for ticker in previous_positions - current_long - current_short:
            self._trades_log.append({
                'date': date,
                'ticker': ticker,
                'side': 'SELL',
                'type': 'CLOSE_LONG'
            })
        
        # New shorts
        for ticker in current_short - previous_positions:
            pred_row = short_stocks[short_stocks['ticker'] == ticker].iloc[0]
            self._trades_log.append({
                'date': date,
                'ticker': ticker,
                'side': 'SHORT',
                'type': 'SHORT',
                'prediction': pred_row['prediction'],
                'forward_return': pred_row['forward_return']
            })
    
    def _create_result(
        self,
        returns_list: List[Dict],
        predictions: pd.DataFrame,
        benchmark: Optional[pd.Series]
    ) -> BacktestResult:
        """Create BacktestResult from simulation data."""
        # Returns series
        returns_df = pd.DataFrame(returns_list)
        returns = returns_df.set_index('date')['return']
        returns.index = pd.to_datetime(returns.index)
        
        # Cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Equity curve
        equity_curve = cumulative * self.config.initial_capital
        
        # Calculate metrics
        metrics = self._calculate_metrics(returns, benchmark)
        
        # Add turnover info
        metrics['avg_turnover'] = returns_df['turnover'].mean()
        metrics['avg_positions'] = returns_df['n_positions'].mean()
        
        # Positions DataFrame
        positions = pd.DataFrame(self._positions_log)
        
        # Trades DataFrame
        trades = pd.DataFrame(self._trades_log) if self._trades_log else pd.DataFrame()
        
        # Add PnL to trades if possible
        if len(trades) > 0 and 'forward_return' in trades.columns:
            # Simple PnL calculation (per unit)
            trades['pnl'] = trades.apply(
                lambda x: x['forward_return'] * self.config.initial_capital / self.config.top_n_long
                if x['side'] in ['BUY', 'LONG'] and pd.notna(x.get('forward_return'))
                else 0,
                axis=1
            )
            trades['return_pct'] = trades['forward_return'].fillna(0)
        
        # Config for reporting
        config_dict = {
            'initial_capital': self.config.initial_capital,
            'top_n_long': self.config.top_n_long,
            'top_n_short': self.config.top_n_short,
            'transaction_cost_bps': self.config.total_cost_bps,
            'rebalance_frequency': self.config.rebalance_frequency,
            'start_date': predictions['date'].min(),
            'end_date': predictions['date'].max(),
            'universe': f"{predictions['ticker'].nunique()} stocks",
            'strategy': f"Long Top {self.config.top_n_long}" + 
                       (f" / Short Bottom {self.config.top_n_short}" if self.config.top_n_short > 0 else "")
        }
        
        return BacktestResult(
            returns=returns,
            cumulative=cumulative,
            equity_curve=equity_curve,
            metrics=metrics,
            positions=positions,
            trades=trades,
            config=config_dict
        )
    
    def _calculate_metrics(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        returns = returns.dropna()
        n = len(returns)
        
        if n < 2:
            return {}
        
        # Determine periods per year based on rebalance frequency
        if self.config.rebalance_frequency == 'daily':
            periods_per_year = 252
        elif self.config.rebalance_frequency == 'weekly':
            periods_per_year = 52
        else:  # monthly
            periods_per_year = 12
        
        # Returns
        total_return = (1 + returns).prod() - 1
        n_years = n / periods_per_year
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        
        # Risk
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Risk-adjusted ratios
        sharpe = annual_return / (volatility + 1e-10)
        
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 0 else 1e-10
        sortino = annual_return / downside_vol
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        calmar = annual_return / (abs(max_drawdown) + 1e-10)
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / (gross_loss + 1e-10)
        
        # VaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95
        
        metrics = {
            # Returns
            'total_return': total_return,
            'annual_return': annual_return,
            'cagr': annual_return,
            'avg_return': returns.mean(),
            'best_period': returns.max(),
            'worst_period': returns.min(),
            
            # Risk
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            
            # Risk-adjusted
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            
            # Trading
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_periods': n,
            'total_trades': len(self._trades_log)
        }
        
        # Benchmark comparison
        if benchmark is not None and len(benchmark) > 0:
            try:
                # Align indices
                common_idx = returns.index.intersection(benchmark.index)
                if len(common_idx) > 1:
                    aligned_returns = returns.loc[common_idx]
                    aligned_benchmark = benchmark.loc[common_idx]
                    
                    excess = aligned_returns - aligned_benchmark
                    tracking_error = excess.std() * np.sqrt(periods_per_year)
                    information_ratio = excess.mean() / (excess.std() + 1e-10) * np.sqrt(periods_per_year)
                    
                    # Beta and Alpha
                    cov_matrix = np.cov(aligned_returns, aligned_benchmark)
                    if cov_matrix.shape == (2, 2) and aligned_benchmark.var() > 1e-10:
                        beta = cov_matrix[0, 1] / aligned_benchmark.var()
                        alpha = aligned_returns.mean() * periods_per_year - beta * aligned_benchmark.mean() * periods_per_year
                    else:
                        beta = 0.0
                        alpha = 0.0
                    
                    metrics['alpha'] = alpha
                    metrics['beta'] = beta
                    metrics['tracking_error'] = tracking_error
                    metrics['information_ratio'] = information_ratio
            except Exception as e:
                logger.warning(f"Benchmark comparison failed: {e}")
        
        return metrics
    
    def _print_header(self):
        """Print backtest header."""
        print("\n" + "="*60)
        print("💼 BACKTESTING")
        print("="*60)
        print(f"   Strategy: Long Top {self.config.top_n_long}" + 
              (f" / Short Bottom {self.config.top_n_short}" if self.config.top_n_short > 0 else ""))
        print(f"   Transaction Cost: {self.config.total_cost_bps:.0f} bps")
        print(f"   Rebalance: {self.config.rebalance_frequency}")
        print(f"   Initial Capital: ${self.config.initial_capital:,.0f}")
    
    def _print_results(self, metrics: Dict):
        """Print backtest results."""
        print(f"\n   {'─'*50}")
        print(f"   📈 BACKTEST RESULTS")
        print(f"   {'─'*50}")
        print(f"   Total Return:      {metrics.get('total_return', 0):>10.1%}")
        print(f"   Annual Return:     {metrics.get('annual_return', 0):>10.1%}")
        print(f"   Volatility:        {metrics.get('volatility', 0):>10.1%}")
        print(f"   Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):>10.2f}")
        print(f"   Sortino Ratio:     {metrics.get('sortino_ratio', 0):>10.2f}")
        print(f"   Max Drawdown:      {metrics.get('max_drawdown', 0):>10.1%}")
        print(f"   Calmar Ratio:      {metrics.get('calmar_ratio', 0):>10.2f}")
        print(f"   Win Rate:          {metrics.get('win_rate', 0):>10.1%}")
        print(f"   Profit Factor:     {metrics.get('profit_factor', 0):>10.2f}")
        print(f"   Avg Turnover:      {metrics.get('avg_turnover', 0):>10.1%}")
        print(f"   Periods:           {metrics.get('n_periods', 0):>10}")
        print(f"   Total Trades:      {metrics.get('total_trades', 0):>10}")
        
        # Risk metrics
        print(f"\n   {'─'*50}")
        print(f"   ⚠️  RISK METRICS")
        print(f"   {'─'*50}")
        print(f"   VaR (95%):         {metrics.get('var_95', 0):>10.2%}")
        print(f"   CVaR (95%):        {metrics.get('cvar_95', 0):>10.2%}")
        
        # Benchmark comparison
        if 'alpha' in metrics:
            print(f"\n   {'─'*50}")
            print(f"   📊 VS BENCHMARK")
            print(f"   {'─'*50}")
            print(f"   Alpha:             {metrics.get('alpha', 0):>10.2%}")
            print(f"   Beta:              {metrics.get('beta', 0):>10.2f}")
            print(f"   Info Ratio:        {metrics.get('information_ratio', 0):>10.2f}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_backtest(
    predictions: pd.DataFrame,
    top_n: int = 10,
    transaction_cost_bps: float = 30,
    rebalance_frequency: str = 'monthly',
    benchmark: Optional[pd.Series] = None,
    verbose: bool = True
) -> Optional[BacktestResult]:
    """
    Quick function to run backtest.
    
    Args:
        predictions: Predictions DataFrame
        top_n: Number of top stocks to hold
        transaction_cost_bps: Transaction cost in bps
        rebalance_frequency: Rebalance frequency
        benchmark: Optional benchmark returns
        verbose: Print results
        
    Returns:
        BacktestResult or None
    
    Example:
        >>> result = run_backtest(predictions, top_n=10, verbose=True)
    """
    config = BacktestConfig(
        top_n_long=top_n,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_frequency=rebalance_frequency
    )
    
    backtester = Backtester(config=config, verbose=verbose)
    return backtester.run(predictions, benchmark)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example with synthetic data
    print("Testing Backtester...")
    
    np.random.seed(42)
    
    # Create sample predictions
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    tickers = [f'STOCK{i}' for i in range(50)]
    
    predictions_list = []
    for date in dates:
        for ticker in tickers:
            predictions_list.append({
                'date': date,
                'ticker': ticker,
                'prediction': np.random.randn(),
                'forward_return': np.random.randn() * 0.05
            })
    
    predictions_df = pd.DataFrame(predictions_list)
    
    # Run backtest
    config = BacktestConfig(
        top_n_long=10,
        transaction_cost_bps=30,
        rebalance_frequency='monthly',
        initial_capital=1_000_000
    )
    
    backtester = Backtester(config=config, verbose=True)
    result = backtester.run(predictions_df)
    
    if result:
        print("\n✅ Backtest completed successfully!")
        print(f"\n📊 Positions logged: {len(result.positions)}")
        print(f"📝 Trades logged: {len(result.trades)}")
        print(result.summary())
    else:
        print("\n❌ Backtest failed!")