"""
Microstructure Factors
======================
Market microstructure and risk-based alpha factors.

These factors capture volatility, liquidity, and volume patterns.

Factors included:
- Volatility (realized volatility)
- Volume Z-Score
- Amihud Illiquidity
- High-Low Range

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


class Volatility(BaseFactor):
    """
    Realized Volatility.
    
    Rolling standard deviation of returns, annualized.
    
    Formula:
        volatility = std(returns) * sqrt(252)
        
    Args:
        window: Lookback period for volatility calculation
        
    Returns:
        Series with annualized volatility values
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(self, window: int):
        if window < 2:
            raise ValueError(f"Volatility window must be >= 2, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"volatility_{self.window}",
            category=FactorCategory.VOLATILITY,
            description=f"{self.window}-day realized volatility (annualized)",
            lookback=self.window + 1,  # Extra day for returns
            higher_is_better=False  # Lower vol often preferred
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute realized volatility."""
        close = df['close']
        
        # Calculate log returns (more stable for volatility)
        log_returns = np.log(close / close.shift(1))
        
        # Rolling standard deviation
        rolling_std = log_returns.rolling(
            window=self.window, 
            min_periods=self.window
        ).std()
        
        # Annualize (252 trading days)
        annualized_vol = rolling_std * np.sqrt(252)
        
        return annualized_vol


class VolatilityRank(BaseFactor):
    """
    Volatility with Cross-Sectional Rank preparation.
    
    Same as Volatility but flagged for ranking.
    
    Args:
        window: Lookback period
        
    Returns:
        Series with volatility values (to be ranked)
    """
    
    REQUIRED_COLUMNS = {'close'}
    
    def __init__(self, window: int):
        if window < 2:
            raise ValueError(f"Volatility window must be >= 2, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"volatility_{self.window}_rank",
            category=FactorCategory.VOLATILITY,
            description=f"{self.window}-day volatility (cross-sectional rank)",
            lookback=self.window + 1,
            is_rank=True,
            higher_is_better=False
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute volatility (ranking happens in registry)."""
        close = df['close']
        log_returns = np.log(close / close.shift(1))
        rolling_std = log_returns.rolling(
            window=self.window,
            min_periods=self.window
        ).std()
        return rolling_std * np.sqrt(252)


class VolumeZScore(BaseFactor):
    """
    Volume Z-Score.
    
    Measures how unusual current volume is compared to recent history.
    High z-score indicates abnormal trading activity.
    
    Formula:
        volume_zscore = (volume - mean_volume) / std_volume
        
    Args:
        window: Lookback period for mean and std
        
    Returns:
        Series with volume z-score values
    """
    
    REQUIRED_COLUMNS = {'close', 'volume'}
    
    def __init__(self, window: int = 21):
        if window < 2:
            raise ValueError(f"Volume window must be >= 2, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"volume_zscore_{self.window}",
            category=FactorCategory.VOLUME,
            description=f"{self.window}-day volume Z-score",
            lookback=self.window,
            higher_is_better=None  # Depends on context
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute volume z-score."""
        volume = df['volume'].astype(float)
        
        # Rolling mean and std
        rolling_mean = volume.rolling(
            window=self.window,
            min_periods=self.window
        ).mean()
        
        rolling_std = volume.rolling(
            window=self.window,
            min_periods=self.window
        ).std()
        
        # Z-score
        zscore = safe_divide(volume - rolling_mean, rolling_std, fill_value=0.0)
        
        return zscore


class AmihudIlliquidity(BaseFactor):
    """
    Amihud Illiquidity Ratio.
    
    Measures price impact per unit of volume traded.
    Higher values = less liquid (more price impact).
    
    Formula:
        illiquidity = mean(|return| / dollar_volume)
        
    Args:
        window: Lookback period for averaging
        
    Returns:
        Series with illiquidity values (log-scaled)
    """
    
    REQUIRED_COLUMNS = {'close', 'volume'}
    
    def __init__(self, window: int = 21):
        if window < 1:
            raise ValueError(f"Amihud window must be >= 1, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"amihud_{self.window}",
            category=FactorCategory.MICROSTRUCTURE,
            description=f"{self.window}-day Amihud illiquidity",
            lookback=self.window + 1,
            higher_is_better=False  # Lower illiquidity preferred
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute Amihud illiquidity."""
        close = df['close']
        volume = df['volume'].astype(float)
        
        # Calculate absolute returns
        abs_return = close.pct_change().abs()
        
        # Dollar volume
        dollar_volume = close * volume
        
        # Daily illiquidity ratio
        daily_illiq = safe_divide(abs_return, dollar_volume, fill_value=0.0)
        
        # Rolling average
        illiquidity = daily_illiq.rolling(
            window=self.window,
            min_periods=self.window
        ).mean()
        
        # Log transform to handle extreme values
        # Add small constant to avoid log(0)
        illiquidity_log = np.log1p(illiquidity * 1e6)  # Scale up before log
        
        return illiquidity_log


class HighLowRange(BaseFactor):
    """
    High-Low Range (Normalized).
    
    Daily trading range as percentage of close.
    Measures intraday volatility.
    
    Formula:
        range = (high - low) / close
        
    Args:
        window: Lookback period for averaging (1 = daily)
        
    Returns:
        Series with range values (percentage)
    """
    
    REQUIRED_COLUMNS = {'high', 'low', 'close'}
    
    def __init__(self, window: int = 1):
        if window < 1:
            raise ValueError(f"Range window must be >= 1, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"hl_range_{self.window}",
            category=FactorCategory.MICROSTRUCTURE,
            description=f"{self.window}-day high-low range",
            lookback=self.window,
            higher_is_better=False  # Lower range = less volatile
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute high-low range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Daily range normalized by close
        daily_range = safe_divide(high - low, close, fill_value=0.0)
        
        # Average over window if needed
        if self.window > 1:
            avg_range = daily_range.rolling(
                window=self.window,
                min_periods=self.window
            ).mean()
            return avg_range
        
        return daily_range


class VolumeRatio(BaseFactor):
    """
    Volume Ratio.
    
    Current volume relative to average volume.
    
    Formula:
        volume_ratio = volume / mean(volume_over_window)
        
    Args:
        window: Lookback period for average
        
    Returns:
        Series with volume ratio values
    """
    
    REQUIRED_COLUMNS = {'volume'}
    
    def __init__(self, window: int = 21):
        if window < 1:
            raise ValueError(f"Volume ratio window must be >= 1, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"volume_ratio_{self.window}",
            category=FactorCategory.VOLUME,
            description=f"Volume relative to {self.window}-day average",
            lookback=self.window,
            higher_is_better=None  # Depends on context
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute volume ratio."""
        volume = df['volume'].astype(float)
        
        # Rolling average (exclude current day)
        avg_volume = volume.shift(1).rolling(
            window=self.window,
            min_periods=self.window
        ).mean()
        
        # Ratio
        ratio = safe_divide(volume, avg_volume, fill_value=1.0)
        
        return ratio


class GarmanKlassVolatility(BaseFactor):
    """
    Garman-Klass Volatility Estimator.
    
    More efficient volatility estimator using OHLC data.
    Uses high-low range and close-open gap for better estimation.
    
    Formula:
        GK = 0.5 * (log(H/L))^2 - (2*log(2) - 1) * (log(C/O))^2
        
    Args:
        window: Lookback period for averaging
        
    Returns:
        Series with GK volatility (annualized)
    """
    
    REQUIRED_COLUMNS = {'open', 'high', 'low', 'close'}
    
    def __init__(self, window: int = 21):
        if window < 1:
            raise ValueError(f"GK window must be >= 1, got {window}")
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"gk_volatility_{self.window}",
            category=FactorCategory.VOLATILITY,
            description=f"{self.window}-day Garman-Klass volatility",
            lookback=self.window,
            higher_is_better=False
        )
    
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """Compute Garman-Klass volatility."""
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        
        # Log ratios
        log_hl = np.log(safe_divide(h, l, fill_value=1.0))
        log_co = np.log(safe_divide(c, o, fill_value=1.0))
        
        # Daily GK variance
        gk_daily = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        
        # Rolling average
        gk_avg = gk_daily.rolling(
            window=self.window,
            min_periods=self.window
        ).mean()
        
        # Annualized volatility (sqrt of variance * sqrt(252))
        gk_vol = np.sqrt(gk_avg) * np.sqrt(252)
        
        return gk_vol


def get_microstructure_factors(
    include_advanced: bool = True
) -> List[BaseFactor]:
    """
    Get all microstructure factors configured in settings.
    
    Args:
        include_advanced: Whether to include advanced estimators
        
    Returns:
        List of BaseFactor instances
    """
    factors = []
    
    # Volatility from config
    for window in settings.features.volatility_windows:
        factors.append(Volatility(window))
        factors.append(VolatilityRank(window))
        logger.debug(f"Added Volatility({window})")
    
    # Volume Z-Score from config
    for window in settings.features.volume_windows:
        factors.append(VolumeZScore(window))
        factors.append(VolumeRatio(window))
        logger.debug(f"Added VolumeZScore({window})")
    
    # Amihud Illiquidity
    factors.append(AmihudIlliquidity(21))
    factors.append(AmihudIlliquidity(63))
    logger.debug("Added Amihud illiquidity")
    
    # High-Low Range
    factors.append(HighLowRange(1))
    factors.append(HighLowRange(5))
    factors.append(HighLowRange(21))
    logger.debug("Added High-Low range")
    
    # Advanced estimators
    if include_advanced:
        factors.append(GarmanKlassVolatility(21))
        factors.append(GarmanKlassVolatility(63))
        logger.debug("Added Garman-Klass volatility")
    
    logger.info(f"Created {len(factors)} microstructure factors")
    
    return factors


def test_microstructure_factors():
    """Test microstructure factors with sample data."""
    print("\n" + "=" * 60)
    print("🧪 TESTING MICROSTRUCTURE FACTORS")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_days = 150
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
    
    # Generate price series
    returns = np.random.randn(n_days) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate volume with some spikes
    base_volume = np.random.randint(1000000, 5000000, n_days)
    volume_spikes = np.random.choice(n_days, 10, replace=False)
    base_volume[volume_spikes] *= 3
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(n_days) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(n_days) * 0.015)),
        'low': prices * (1 - np.abs(np.random.randn(n_days) * 0.015)),
        'close': prices,
        'volume': base_volume
    })
    
    # Fix high < low issues
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1) * 1.001
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1) * 0.999
    
    print(f"\n📊 Sample data: {len(df)} rows")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   Volume range: {df['volume'].min():,.0f} - {df['volume'].max():,.0f}")
    
    # Test each factor
    factors = get_microstructure_factors()
    
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
    test_microstructure_factors()