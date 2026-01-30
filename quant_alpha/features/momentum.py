"""
Momentum Factors
================
Trend-following alpha factors based on price momentum.

These factors capture the tendency of trending assets
to continue moving in the same direction.

Factors included:
- Momentum (simple return over window)
- Momentum Rank (cross-sectional rank of momentum)
- Rate of Change (ROC)
- MACD

Author: [Your Name]
Last Updated: 2024
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

try:
    from quant_alpha.features.base import (
        BaseFactor,
        FactorInfo,
        FactorCategory,
        safe_divide,
    )
    from config.settings import settings
except ImportError:
    from .base import (
        BaseFactor,
        FactorInfo,
        FactorCategory,
        safe_divide,
    )
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import settings


logger = logging.getLogger(__name__)


class Momentum(BaseFactor):
    """
    Simple Price Momentum.
    
    Returns over a specified lookback period.
    
    Formula:
        momentum = (close / close_n_days_ago) - 1
        
    Args:
        window: Lookback period in trading days
        
    Returns:
        Series with momentum values (percentage returns)
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(self, window: int):
        if window < 1:
            raise ValueError(f"Momentum window must be >= 1, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"mom_{self.window}",
            category=FactorCategory.MOMENTUM,
            description=f"{self.window}-day momentum (return)",
            lookback=self.window,
            higher_is_better=True  # Higher momentum = stronger signal
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute momentum."""
        close = df['close']
        
        # Calculate return over window
        momentum = close.pct_change(periods=self.window)
        
        return momentum


class MomentumRank(BaseFactor):
    """
    Momentum with Cross-Sectional Rank preparation.
    
    Same as Momentum but flagged as rank factor.
    Cross-sectional ranking happens in the registry.
    
    Args:
        window: Lookback period in trading days
        
    Returns:
        Series with momentum values (to be ranked cross-sectionally)
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(self, window: int):
        if window < 1:
            raise ValueError(f"Momentum window must be >= 1, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"mom_{self.window}_rank",
            category=FactorCategory.MOMENTUM,
            description=f"{self.window}-day momentum (cross-sectional rank)",
            lookback=self.window,
            is_rank=True,
            higher_is_better=True
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute momentum (ranking happens in registry)."""
        close = df['close']
        momentum = close.pct_change(periods=self.window)
        return momentum


class RateOfChange(BaseFactor):
    """
    Rate of Change (ROC).
    
    Percentage change from N periods ago.
    Similar to momentum but expressed differently.
    
    Formula:
        ROC = ((close - close_n) / close_n) * 100
        
    Args:
        window: Lookback period
        
    Returns:
        Series with ROC values (percentage)
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(self, window: int = 10):
        if window < 1:
            raise ValueError(f"ROC window must be >= 1, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"roc_{self.window}",
            category=FactorCategory.MOMENTUM,
            description=f"{self.window}-day Rate of Change",
            lookback=self.window,
            higher_is_better=True
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute ROC."""
        close = df['close']
        close_lagged = close.shift(self.window)
        
        roc = safe_divide(close - close_lagged, close_lagged, fill_value=0.0) * 100
        
        return roc


class MACD(BaseFactor):
    """
    Moving Average Convergence Divergence (MACD).
    
    Difference between short and long EMAs.
    Positive MACD indicates bullish momentum.
    
    Formula:
        MACD = EMA(short) - EMA(long)
        
    Args:
        short_window: Short EMA period (default: 12)
        long_window: Long EMA period (default: 26)
        
    Returns:
        Series with MACD values
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(self, short_window: int = 12, long_window: int = 26):
        if short_window >= long_window:
            raise ValueError(
                f"short_window ({short_window}) must be < long_window ({long_window})"
            )
        self.short_window = short_window
        self.long_window = long_window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"macd_{self.short_window}_{self.long_window}",
            category=FactorCategory.MOMENTUM,
            description=f"MACD ({self.short_window}/{self.long_window})",
            lookback=self.long_window,
            higher_is_better=True
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute MACD."""
        close = df['close']
        
        # Calculate EMAs
        ema_short = close.ewm(span=self.short_window, adjust=False).mean()
        ema_long = close.ewm(span=self.long_window, adjust=False).mean()
        
        # MACD line
        macd = ema_short - ema_long
        
        # Normalize by price level for comparability
        macd_normalized = safe_divide(macd, close, fill_value=0.0)
        
        return macd_normalized


class MACDSignal(BaseFactor):
    """
    MACD Signal Line Crossover.
    
    Difference between MACD and its signal line.
    Positive = bullish crossover, Negative = bearish.
    
    Args:
        short_window: Short EMA period
        long_window: Long EMA period
        signal_window: Signal line EMA period
        
    Returns:
        Series with MACD histogram values
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(
        self, 
        short_window: int = 12, 
        long_window: int = 26, 
        signal_window: int = 9
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"macd_signal_{self.short_window}_{self.long_window}_{self.signal_window}",
            category=FactorCategory.MOMENTUM,
            description=f"MACD Signal ({self.short_window}/{self.long_window}/{self.signal_window})",
            lookback=self.long_window + self.signal_window,
            higher_is_better=True
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute MACD histogram."""
        close = df['close']
        
        # Calculate EMAs
        ema_short = close.ewm(span=self.short_window, adjust=False).mean()
        ema_long = close.ewm(span=self.long_window, adjust=False).mean()
        
        # MACD line
        macd = ema_short - ema_long
        
        # Signal line (EMA of MACD)
        signal = macd.ewm(span=self.signal_window, adjust=False).mean()
        
        # Histogram (MACD - Signal)
        histogram = macd - signal
        
        # Normalize by price level
        histogram_normalized = safe_divide(histogram, close, fill_value=0.0)
        
        return histogram_normalized


class MomentumAcceleration(BaseFactor):
    """
    Momentum Acceleration.
    
    Change in momentum (second derivative of price).
    Positive = momentum is increasing.
    
    Formula:
        accel = mom(t) - mom(t-n)
        
    Args:
        window: Momentum calculation window
        
    Returns:
        Series with momentum acceleration values
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(self, window: int = 21):
        if window < 2:
            raise ValueError(f"Window must be >= 2, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"mom_accel_{self.window}",
            category=FactorCategory.MOMENTUM,
            description=f"{self.window}-day momentum acceleration",
            lookback=self.window * 2,
            higher_is_better=True
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute momentum acceleration."""
        close = df['close']
        
        # Current momentum
        mom_current = close.pct_change(periods=self.window)
        
        # Previous momentum
        mom_previous = mom_current.shift(self.window)
        
        # Acceleration
        acceleration = mom_current - mom_previous
        
        return acceleration


def get_momentum_factors(include_macd: bool = True) -> List[BaseFactor]:
    """
    Get all momentum factors configured in settings.
    
    Args:
        include_macd: Whether to include MACD indicators
        
    Returns:
        List of BaseFactor instances
    """
    factors = []
    
    # Simple momentum from config
    for window in settings.features.momentum_windows:
        factors.append(Momentum(window))
        factors.append(MomentumRank(window))
        logger.debug(f"Added Momentum({window}) and MomentumRank({window})")
    
    # Rate of Change
    roc_windows = [5, 10, 21]
    for window in roc_windows:
        factors.append(RateOfChange(window))
        logger.debug(f"Added ROC({window})")
    
    # MACD indicators
    if include_macd:
        factors.append(MACD(12, 26))
        factors.append(MACDSignal(12, 26, 9))
        logger.debug("Added MACD indicators")
    
    # Momentum acceleration
    factors.append(MomentumAcceleration(21))
    logger.debug("Added MomentumAcceleration")
    
    logger.info(f"Created {len(factors)} momentum factors")
    
    return factors


def test_momentum_factors():
    """Test momentum factors with sample data."""
    print("\n" + "=" * 60)
    print("🧪 TESTING MOMENTUM FACTORS")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_days = 150
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
    
    # Generate trending price series
    trend = np.linspace(0, 0.3, n_days)  # 30% uptrend
    noise = np.random.randn(n_days) * 0.02
    returns = trend / n_days + noise
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(n_days) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(n_days) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(n_days) * 0.01)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    print(f"\n📊 Sample data: {len(df)} rows (trending up)")
    print(f"   Price: ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
    
    # Test each factor
    factors = get_momentum_factors()
    
    print(f"\n📈 Testing {len(factors)} factors:")
    
    for factor in factors:
        try:
            result = factor.compute(df)
            valid_pct = (result.notna().sum() / len(result)) * 100
            last_value = result.dropna().iloc[-1] if result.notna().any() else np.nan
            print(f"   ✅ {factor.info.name}: {valid_pct:.0f}% valid, last={last_value:.4f}")
        except Exception as e:
            print(f"   ❌ {factor.info.name}: {e}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    test_momentum_factors()