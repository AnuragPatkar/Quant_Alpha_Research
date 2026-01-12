"""
Performance Metrics
===================
Additional performance and risk metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


class PerformanceMetrics:
    """Extended performance metrics calculations."""
    
    @staticmethod
    def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio vs benchmark."""
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
        return returns[returns <= var].mean()
    
    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        return gains / (losses + 1e-10)
    
    @staticmethod
    def get_extended_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Get comprehensive performance metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        metrics['var_95'] = PerformanceMetrics.value_at_risk(returns, 0.05)
        metrics['cvar_95'] = PerformanceMetrics.conditional_var(returns, 0.05)
        metrics['tail_ratio'] = PerformanceMetrics.tail_ratio(returns)
        metrics['omega_ratio'] = PerformanceMetrics.omega_ratio(returns)
        metrics['max_dd_duration'] = PerformanceMetrics.maximum_drawdown_duration(returns)
        
        # Benchmark comparison
        if benchmark_returns is not None:
            metrics['information_ratio'] = PerformanceMetrics.information_ratio(returns, benchmark_returns)
            metrics['beta'] = np.cov(returns, benchmark_returns)[0, 1] / (benchmark_returns.var() + 1e-10)
            metrics['alpha'] = returns.mean() - metrics['beta'] * benchmark_returns.mean()
        
        return metrics


class RiskMetrics:
    """Risk-specific metrics and analysis."""
    
    @staticmethod
    def rolling_sharpe(returns: pd.Series, window: int = 12) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        return rolling_mean / (rolling_std + 1e-10)
    
    @staticmethod
    def rolling_volatility(returns: pd.Series, window: int = 12, annualize: bool = True) -> pd.Series:
        """Calculate rolling volatility."""
        vol = returns.rolling(window).std()
        if annualize:
            vol *= np.sqrt(12)  # Assuming monthly data
        return vol
    
    @staticmethod
    def drawdown_analysis(returns: pd.Series) -> Dict[str, float]:
        """Comprehensive drawdown analysis."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        return {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'drawdown_periods': (drawdown < -0.01).sum(),  # Periods with >1% drawdown
            'recovery_factor': abs(returns.sum() / drawdown.min()) if drawdown.min() < 0 else np.inf
        }
    
    @staticmethod
    def risk_adjusted_returns(returns: pd.Series) -> Dict[str, float]:
        """Calculate various risk-adjusted return metrics."""
        annual_return = (1 + returns).prod() ** (12 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(12)
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 1e-10
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())
        
        return {
            'sharpe_ratio': annual_return / (volatility + 1e-10),
            'sortino_ratio': annual_return / downside_vol,
            'calmar_ratio': annual_return / (max_dd + 1e-10),
            'sterling_ratio': annual_return / (drawdown[drawdown < 0].std() + 1e-10) if (drawdown < 0).any() else 0
        }