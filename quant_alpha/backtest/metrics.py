"""
Performance Metrics
===================
Comprehensive performance and risk metrics calculations.

Classes:
    - PerformanceMetrics: Return and performance calculations
    - RiskMetrics: Risk-adjusted and downside metrics

Author: Senior Quant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class PerformanceMetrics:
    """
    Extended performance metrics calculations.
    
    Can be used as:
        1. Static methods for individual calculations
        2. Instance methods for comprehensive analysis
    
    Example (static):
        >>> ir = PerformanceMetrics.information_ratio(returns, benchmark)
        >>> var = PerformanceMetrics.value_at_risk(returns, 0.05)
    
    Example (instance):
        >>> pm = PerformanceMetrics(returns, benchmark)
        >>> all_metrics = pm.calculate_all()
    """
    
    def __init__(
        self,
        returns: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 12  # Monthly data
    ):
        """
        Initialize performance metrics calculator.
        
        Args:
            returns: Returns series
            benchmark_returns: Optional benchmark returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Periods per year (252=daily, 12=monthly)
        """
        self.returns = returns.dropna() if returns is not None else None
        self.benchmark = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.rf_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
        if self.returns is not None:
            self.rf_period = (1 + self.rf_rate) ** (1 / self.periods_per_year) - 1
    
    def calculate_all(self) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Returns:
            Dictionary with all metrics
        """
        if self.returns is None or len(self.returns) < 2:
            return {}
        
        metrics = {}
        
        # Basic returns
        metrics.update(self._return_metrics())
        
        # Risk-adjusted
        metrics.update(self._risk_adjusted_metrics())
        
        # Drawdown
        metrics.update(self._drawdown_metrics())
        
        # Trading
        metrics.update(self._trading_metrics())
        
        # Extended metrics
        metrics.update(self.get_extended_metrics(self.returns, self.benchmark))
        
        return metrics
    
    def _return_metrics(self) -> Dict[str, float]:
        """Calculate return metrics."""
        total_return = (1 + self.returns).prod() - 1
        n_periods = len(self.returns)
        n_years = n_periods / self.periods_per_year
        
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'annualized_return': cagr,
            'avg_return': self.returns.mean(),
            'best_period': self.returns.max(),
            'worst_period': self.returns.min(),
            'n_periods': n_periods
        }
    
    def _risk_adjusted_metrics(self) -> Dict[str, float]:
        """Calculate risk-adjusted metrics."""
        volatility = self.returns.std() * np.sqrt(self.periods_per_year)
        
        excess_returns = self.returns - self.rf_period
        sharpe = excess_returns.mean() / (self.returns.std() + 1e-10) * np.sqrt(self.periods_per_year)
        
        downside = self.returns[self.returns < self.rf_period]
        downside_std = downside.std() * np.sqrt(self.periods_per_year) if len(downside) > 1 else 1e-10
        sortino = excess_returns.mean() / (downside.std() + 1e-10) * np.sqrt(self.periods_per_year)
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'downside_deviation': downside_std
        }
    
    def _drawdown_metrics(self) -> Dict[str, float]:
        """Calculate drawdown metrics."""
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        max_dd = drawdown.min()
        cagr = self._return_metrics()['cagr']
        calmar = cagr / (abs(max_dd) + 1e-10)
        
        return {
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'max_dd_duration': self.maximum_drawdown_duration(self.returns)
        }
    
    def _trading_metrics(self) -> Dict[str, float]:
        """Calculate trading metrics."""
        wins = (self.returns > 0).sum()
        losses = (self.returns < 0).sum()
        total = wins + losses
        
        win_rate = wins / total if total > 0 else 0.0
        
        gross_profits = self.returns[self.returns > 0].sum()
        gross_losses = abs(self.returns[self.returns < 0].sum())
        profit_factor = gross_profits / (gross_losses + 1e-10)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': self.returns[self.returns > 0].mean() if wins > 0 else 0.0,
            'avg_loss': self.returns[self.returns < 0].mean() if losses > 0 else 0.0
        }
    
    # =========================================================================
    # STATIC METHODS (Original API preserved)
    # =========================================================================
    
    @staticmethod
    def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio vs benchmark."""
        if len(returns) != len(benchmark_returns):
            # Align indices
            common_idx = returns.index.intersection(benchmark_returns.index)
            returns = returns.loc[common_idx]
            benchmark_returns = benchmark_returns.loc[common_idx]
        
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std()
        return excess_returns.mean() / (tracking_error + 1e-10)
    
    @staticmethod
    def maximum_drawdown_duration(returns: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        # Find drawdown periods
        in_drawdown = drawdown < -0.001  # More than 0.1% drawdown
        
        if not in_drawdown.any():
            return 0
        
        # Calculate consecutive drawdown periods
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        # Add final period if still in drawdown
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    @staticmethod
    def tail_ratio(returns: pd.Series, percentile: float = 0.05) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        upper = returns.quantile(1 - percentile)
        lower = returns.quantile(percentile)
        return abs(upper / (lower + 1e-10))
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Value at Risk."""
        return returns.quantile(confidence)
    
    @staticmethod
    def conditional_var(returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = PerformanceMetrics.value_at_risk(returns, confidence)
        tail_returns = returns[returns <= var]
        return tail_returns.mean() if len(tail_returns) > 0 else var
    
    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        return gains / (losses + 1e-10)
    
    @staticmethod
    def get_extended_metrics(
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Get comprehensive performance metrics."""
        metrics = {}
        
        returns = returns.dropna()
        
        if len(returns) < 2:
            return metrics
        
        # Basic metrics
        metrics['skewness'] = stats.skew(returns) if len(returns) > 2 else 0.0
        metrics['kurtosis'] = stats.kurtosis(returns) if len(returns) > 3 else 0.0
        metrics['var_95'] = PerformanceMetrics.value_at_risk(returns, 0.05)
        metrics['cvar_95'] = PerformanceMetrics.conditional_var(returns, 0.05)
        metrics['tail_ratio'] = PerformanceMetrics.tail_ratio(returns)
        metrics['omega_ratio'] = PerformanceMetrics.omega_ratio(returns)
        metrics['max_dd_duration'] = PerformanceMetrics.maximum_drawdown_duration(returns)
        
        # Benchmark comparison
        if benchmark_returns is not None and len(benchmark_returns) > 1:
            benchmark_returns = benchmark_returns.dropna()
            
            # Align indices
            common_idx = returns.index.intersection(benchmark_returns.index)
            if len(common_idx) > 1:
                aligned_returns = returns.loc[common_idx]
                aligned_benchmark = benchmark_returns.loc[common_idx]
                
                metrics['information_ratio'] = PerformanceMetrics.information_ratio(
                    aligned_returns, aligned_benchmark
                )
                
                # Beta and Alpha
                cov_matrix = np.cov(aligned_returns, aligned_benchmark)
                if cov_matrix.shape == (2, 2):
                    metrics['beta'] = cov_matrix[0, 1] / (aligned_benchmark.var() + 1e-10)
                    metrics['alpha'] = aligned_returns.mean() - metrics['beta'] * aligned_benchmark.mean()
                else:
                    metrics['beta'] = 0.0
                    metrics['alpha'] = 0.0
        
        return metrics


class RiskMetrics:
    """
    Risk-specific metrics and analysis.
    
    Provides:
        - Rolling risk metrics
        - Drawdown analysis
        - Risk-adjusted return calculations
    
    Example:
        >>> rm = RiskMetrics(returns)
        >>> risk_stats = rm.calculate_all()
        >>> rolling_vol = rm.rolling_volatility(window=12)
    """
    
    def __init__(
        self,
        returns: Optional[pd.Series] = None,
        periods_per_year: int = 12
    ):
        """
        Initialize risk metrics calculator.
        
        Args:
            returns: Returns series
            periods_per_year: Periods per year
        """
        self.returns = returns.dropna() if returns is not None else None
        self.periods_per_year = periods_per_year
    
    def calculate_all(self) -> Dict[str, float]:
        """
        Calculate all risk metrics.
        
        Returns:
            Dictionary with all risk metrics
        """
        if self.returns is None or len(self.returns) < 2:
            return {}
        
        metrics = {}
        
        # Drawdown analysis
        metrics.update(self.drawdown_analysis(self.returns))
        
        # Risk-adjusted returns
        metrics.update(self.risk_adjusted_returns(self.returns))
        
        # VaR metrics
        metrics['var_95'] = PerformanceMetrics.value_at_risk(self.returns, 0.05)
        metrics['var_99'] = PerformanceMetrics.value_at_risk(self.returns, 0.01)
        metrics['cvar_95'] = PerformanceMetrics.conditional_var(self.returns, 0.05)
        
        return metrics
    
    # =========================================================================
    # STATIC METHODS (Original API preserved)
    # =========================================================================
    
    @staticmethod
    def rolling_sharpe(returns: pd.Series, window: int = 12) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        return rolling_mean / (rolling_std + 1e-10)
    
    @staticmethod
    def rolling_volatility(
        returns: pd.Series,
        window: int = 12,
        annualize: bool = True,
        periods_per_year: int = 12
    ) -> pd.Series:
        """Calculate rolling volatility."""
        vol = returns.rolling(window).std()
        if annualize:
            vol *= np.sqrt(periods_per_year)
        return vol
    
    @staticmethod
    def drawdown_analysis(returns: pd.Series) -> Dict[str, float]:
        """Comprehensive drawdown analysis."""
        returns = returns.dropna()
        
        if len(returns) < 2:
            return {'max_drawdown': 0, 'avg_drawdown': 0, 'drawdown_periods': 0, 'recovery_factor': 0}
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        max_dd = drawdown.min()
        in_dd = drawdown < -0.01  # >1% drawdown
        
        return {
            'max_drawdown': max_dd,
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'drawdown_periods': int(in_dd.sum()),
            'recovery_factor': abs(returns.sum() / max_dd) if max_dd < 0 else np.inf
        }
    
    @staticmethod
    def risk_adjusted_returns(
        returns: pd.Series,
        periods_per_year: int = 12
    ) -> Dict[str, float]:
        """Calculate various risk-adjusted return metrics."""
        returns = returns.dropna()
        
        if len(returns) < 2:
            return {'sharpe_ratio': 0, 'sortino_ratio': 0, 'calmar_ratio': 0, 'sterling_ratio': 0}
        
        n_periods = len(returns)
        n_years = n_periods / periods_per_year
        
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 1 else 1e-10
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())
        
        # Sterling ratio (return / avg drawdown)
        avg_dd = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 1e-10
        
        return {
            'sharpe_ratio': annual_return / (volatility + 1e-10),
            'sortino_ratio': annual_return / downside_vol,
            'calmar_ratio': annual_return / (max_dd + 1e-10),
            'sterling_ratio': annual_return / avg_dd
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_metrics(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 12
) -> Dict[str, float]:
    """
    Calculate all metrics (convenience function).
    
    Args:
        returns: Returns series
        benchmark: Optional benchmark
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year
        
    Returns:
        Dictionary with all metrics
    """
    perf = PerformanceMetrics(returns, benchmark, risk_free_rate, periods_per_year)
    risk = RiskMetrics(returns, periods_per_year)
    
    metrics = perf.calculate_all()
    metrics.update(risk.calculate_all())
    
    return metrics


def print_metrics_summary(metrics: Dict[str, float], title: str = "Performance Summary"):
    """
    Print formatted metrics summary.
    
    Args:
        metrics: Metrics dictionary
        title: Title for summary
    """
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)
    
    # Returns
    print("\n  RETURNS")
    print("  " + "-"*50)
    print(f"  Total Return:      {metrics.get('total_return', 0):>10.2%}")
    print(f"  CAGR:              {metrics.get('cagr', 0):>10.2%}")
    print(f"  Avg Return:        {metrics.get('avg_return', 0):>10.4%}")
    
    # Risk-Adjusted
    print("\n  RISK-ADJUSTED")
    print("  " + "-"*50)
    print(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):>10.2f}")
    print(f"  Sortino Ratio:     {metrics.get('sortino_ratio', 0):>10.2f}")
    print(f"  Calmar Ratio:      {metrics.get('calmar_ratio', 0):>10.2f}")
    print(f"  Omega Ratio:       {metrics.get('omega_ratio', 0):>10.2f}")
    
    # Risk
    print("\n  RISK")
    print("  " + "-"*50)
    print(f"  Volatility:        {metrics.get('volatility', 0):>10.2%}")
    print(f"  Max Drawdown:      {metrics.get('max_drawdown', 0):>10.2%}")
    print(f"  VaR (95%):         {metrics.get('var_95', 0):>10.2%}")
    print(f"  CVaR (95%):        {metrics.get('cvar_95', 0):>10.2%}")
    
    # Trading
    print("\n  TRADING")
    print("  " + "-"*50)
    print(f"  Win Rate:          {metrics.get('win_rate', 0):>10.2%}")
    print(f"  Profit Factor:     {metrics.get('profit_factor', 0):>10.2f}")
    
    # Distribution
    if 'skewness' in metrics:
        print("\n  DISTRIBUTION")
        print("  " + "-"*50)
        print(f"  Skewness:          {metrics.get('skewness', 0):>10.4f}")
        print(f"  Kurtosis:          {metrics.get('kurtosis', 0):>10.4f}")
    
    # Benchmark
    if 'alpha' in metrics:
        print("\n  VS BENCHMARK")
        print("  " + "-"*50)
        print(f"  Alpha:             {metrics.get('alpha', 0):>10.4f}")
        print(f"  Beta:              {metrics.get('beta', 0):>10.2f}")
        print(f"  Info Ratio:        {metrics.get('information_ratio', 0):>10.2f}")
    
    print("\n" + "="*60 + "\n")