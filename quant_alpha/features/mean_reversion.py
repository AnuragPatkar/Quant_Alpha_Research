"""
Mean Reversion Factors
======================
Counter-trend alpha factors based on mean reversion principles.

These factors identify overbought/oversold conditions and
predict price reversals toward historical averages.

Factors included:
- RSI (Relative Strength Index)
- Distance from Moving Average
- Z-Score
- Bollinger Band Position

Author: [Your Name]
Last Updated: 2024
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

# Proper imports (no sys.path hacking)
try:
    from quant_alpha.features.base import (
        BaseFactor, 
        FactorInfo, 
        FactorCategory,
        safe_divide,
        clip_values,
    )
    from config.settings import settings
except ImportError:
    from .base import (
        BaseFactor, 
        FactorInfo, 
        FactorCategory,
        safe_divide,
        clip_values,
    )
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import settings


logger = logging.getLogger(__name__)


class RSI(BaseFactor):
    """
    Relative Strength Index (RSI).
    
    Measures momentum as ratio of up moves to total moves.
    - RSI > 70: Overbought (expect reversal down)
    - RSI < 30: Oversold (expect reversal up)
    
    Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
    
    Args:
        window: Lookback period (default: 14)
        
    Returns:
        Series with RSI values (0-100)
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(self, window: int = 14):
        if window < 2:
            raise ValueError(f"RSI window must be >= 2, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"rsi_{self.window}",  # Consistent naming (no 'd' suffix)
            category=FactorCategory.MEAN_REVERSION,
            description=f"{self.window}-day Relative Strength Index",
            lookback=self.window + 1,  # Need extra day for diff
            higher_is_better=False  # Lower RSI = better buy signal
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute RSI values."""
        close = df['close']
        
        # Calculate price changes
        delta = close.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)
        
        # Calculate average gains and losses using EMA (Wilder's smoothing)
        # This is more accurate than simple moving average
        avg_gain = gains.ewm(
            alpha=1/self.window, 
            min_periods=self.window,
            adjust=False
        ).mean()
        
        avg_loss = losses.ewm(
            alpha=1/self.window,
            min_periods=self.window,
            adjust=False
        ).mean()
        
        # Calculate RS and RSI using safe division
        rs = safe_divide(avg_gain, avg_loss, fill_value=0.0)
        rsi = 100 - safe_divide(100, 1 + rs, fill_value=50)
        
        # Clip to valid range (should already be 0-100, but just in case)
        rsi = clip_values(rsi, lower=0, upper=100)
        
        return rsi


class DistanceFromMA(BaseFactor):
    """
    Distance from Moving Average.
    
    Measures how far current price is from its moving average.
    Expressed as percentage deviation.
    
    - Positive: Price above MA (potentially overbought)
    - Negative: Price below MA (potentially oversold)
    
    Formula:
        dist_ma = (close - MA) / MA
        
    Args:
        window: Moving average period
        
    Returns:
        Series with percentage distance from MA
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(self, window: int):
        if window < 1:
            raise ValueError(f"MA window must be >= 1, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"dist_ma_{self.window}",  # Consistent naming
            category=FactorCategory.MEAN_REVERSION,
            description=f"Distance from {self.window}-day moving average",
            lookback=self.window,
            higher_is_better=False  # Below MA = better buy signal
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute distance from MA."""
        close = df['close']
        
        # Calculate moving average
        ma = close.rolling(window=self.window, min_periods=self.window).mean()
        
        # Calculate percentage distance using safe division
        distance = safe_divide(close - ma, ma, fill_value=0.0)
        
        return distance


class ZScore(BaseFactor):
    """
    Price Z-Score.
    
    Number of standard deviations from the mean.
    Useful for identifying extreme price levels.
    
    - Z > 2: Price is 2+ std above mean (overbought)
    - Z < -2: Price is 2+ std below mean (oversold)
    
    Formula:
        zscore = (close - mean) / std
        
    Args:
        window: Lookback period for mean and std
        
    Returns:
        Series with z-score values
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(self, window: int = 21):
        if window < 2:
            raise ValueError(f"Z-score window must be >= 2, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"zscore_{self.window}",
            category=FactorCategory.MEAN_REVERSION,
            description=f"{self.window}-day price Z-score",
            lookback=self.window,
            higher_is_better=False
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute z-score."""
        close = df['close']
        
        # Calculate rolling mean and std
        rolling_mean = close.rolling(
            window=self.window, 
            min_periods=self.window
        ).mean()
        
        rolling_std = close.rolling(
            window=self.window, 
            min_periods=self.window
        ).std()
        
        # Calculate z-score using safe division
        zscore = safe_divide(close - rolling_mean, rolling_std, fill_value=0.0)
        
        return zscore


class BollingerPosition(BaseFactor):
    """
    Position within Bollinger Bands.
    
    Normalized position between lower and upper Bollinger Bands.
    
    - Value near 0: Price near lower band (oversold)
    - Value near 1: Price near upper band (overbought)
    - Value < 0 or > 1: Price outside bands (extreme)
    
    Formula:
        position = (close - lower_band) / (upper_band - lower_band)
        
    Args:
        window: Period for MA and std calculation
        num_std: Number of standard deviations for bands (default: 2)
        
    Returns:
        Series with Bollinger position (can be outside 0-1)
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        if window < 2:
            raise ValueError(f"Bollinger window must be >= 2, got {window}")
        if num_std <= 0:
            raise ValueError(f"num_std must be > 0, got {num_std}")
        
        self.window = window
        self.num_std = num_std
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"bb_position_{self.window}",
            category=FactorCategory.MEAN_REVERSION,
            description=f"Bollinger Band position ({self.window}-day, {self.num_std}σ)",
            lookback=self.window,
            higher_is_better=False
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute Bollinger Band position."""
        close = df['close']
        
        # Calculate MA and std
        ma = close.rolling(window=self.window, min_periods=self.window).mean()
        std = close.rolling(window=self.window, min_periods=self.window).std()
        
        # Calculate bands
        upper = ma + self.num_std * std
        lower = ma - self.num_std * std
        band_width = upper - lower
        
        # Calculate position using safe division
        position = safe_divide(close - lower, band_width, fill_value=0.5)
        
        return position


class MeanReversionStrength(BaseFactor):
    """
    Mean Reversion Strength indicator.
    
    Combines multiple mean reversion signals into single score.
    Uses RSI, distance from MA, and z-score.
    
    Args:
        rsi_window: Window for RSI calculation
        ma_window: Window for MA distance calculation
        zscore_window: Window for z-score calculation
        
    Returns:
        Series with composite mean reversion score (-1 to 1)
        Negative = oversold, Positive = overbought
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(
        self, 
        rsi_window: int = 14, 
        ma_window: int = 21, 
        zscore_window: int = 21
    ):
        self.rsi_window = rsi_window
        self.ma_window = ma_window
        self.zscore_window = zscore_window
        
        # Component factors
        self._rsi = RSI(rsi_window)
        self._dist_ma = DistanceFromMA(ma_window)
        self._zscore = ZScore(zscore_window)
    
    @property
    def info(self) -> FactorInfo:
        max_lookback = max(self.rsi_window, self.ma_window, self.zscore_window)
        return FactorInfo(
            name="mean_reversion_strength",
            category=FactorCategory.MEAN_REVERSION,
            description="Composite mean reversion indicator",
            lookback=max_lookback + 1,
            higher_is_better=False
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute composite mean reversion score."""
        # Get component signals
        rsi = self._rsi.compute(df)
        dist_ma = self._dist_ma.compute(df)
        zscore = self._zscore.compute(df)
        
        # Normalize RSI to -1 to 1 scale
        # RSI 50 = neutral, RSI 0 = -1, RSI 100 = 1
        rsi_normalized = (rsi - 50) / 50
        
        # Normalize z-score to roughly -1 to 1 (clip at ±3)
        zscore_normalized = zscore.clip(-3, 3) / 3
        
        # Normalize distance from MA (clip at ±20%)
        dist_ma_normalized = dist_ma.clip(-0.2, 0.2) / 0.2
        
        # Combine with equal weights
        composite = (rsi_normalized + zscore_normalized + dist_ma_normalized) / 3
        
        return composite


def get_mean_reversion_factors(
    include_composite: bool = True
) -> List[BaseFactor]:
    """
    Get all mean reversion factors configured in settings.
    
    Args:
        include_composite: Whether to include composite indicators
        
    Returns:
        List of BaseFactor instances
    """
    factors = []
    
    # RSI factors from config
    for window in settings.features.rsi_windows:
        factors.append(RSI(window))
        logger.debug(f"Added RSI({window})")
    
    # Distance from MA factors from config
    for window in settings.features.ma_windows:
        factors.append(DistanceFromMA(window))
        logger.debug(f"Added DistanceFromMA({window})")
    
    # Z-score factors (commonly used windows)
    zscore_windows = [10, 21, 63]  # 2 weeks, 1 month, 3 months
    for window in zscore_windows:
        factors.append(ZScore(window))
        logger.debug(f"Added ZScore({window})")
    
    # Bollinger Band position
    bb_windows = [20, 50]
    for window in bb_windows:
        factors.append(BollingerPosition(window))
        logger.debug(f"Added BollingerPosition({window})")
    
    # Composite indicator
    if include_composite:
        factors.append(MeanReversionStrength())
        logger.debug("Added MeanReversionStrength")
    
    logger.info(f"Created {len(factors)} mean reversion factors")
    
    return factors


# Utility function for testing
def test_mean_reversion_factors():
    """Test mean reversion factors with sample data."""
    print("\n" + "=" * 60)
    print("🧪 TESTING MEAN REVERSION FACTORS")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
    
    # Generate realistic price series with mean reversion
    returns = np.random.randn(n_days) * 0.02  # 2% daily vol
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(n_days) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(n_days) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(n_days) * 0.01)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    print(f"\n📊 Sample data: {len(df)} rows")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Test each factor
    factors = get_mean_reversion_factors()
    
    print(f"\n📈 Testing {len(factors)} factors:")
    
    for factor in factors:
        try:
            result = factor.compute(df)
            valid_pct = (result.notna().sum() / len(result)) * 100
            print(f"   ✅ {factor.info.name}: {valid_pct:.0f}% valid values")
            print(f"      Range: [{result.min():.3f}, {result.max():.3f}]")
        except Exception as e:
            print(f"   ❌ {factor.info.name}: {e}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    test_mean_reversion_factors()