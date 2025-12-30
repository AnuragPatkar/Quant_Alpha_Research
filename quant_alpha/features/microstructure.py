"""
Microstructure & Volatility Factors
===================================
Volume, liquidity, and volatility factors.
(Important for HFT firms!)
"""

import pandas as pd
import numpy as np
from typing import List
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_alpha.features.base import BaseFactor, FactorInfo, FactorCategory
from config.settings import settings


class Volatility(BaseFactor):
    """Historical volatility (annualized)."""
    
    def __init__(self, window: int):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"volatility_{self.window}d",
            category=FactorCategory.VOLATILITY,
            description=f"{self.window}-day volatility (annualized)",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df['close'].pct_change()
        return returns.rolling(self.window).std() * np.sqrt(252)


class ATR(BaseFactor):
    """Average True Range (normalized by price)."""
    
    def __init__(self, window: int = 14):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"atr_{self.window}d",
            category=FactorCategory.VOLATILITY,
            description=f"{self.window}-day ATR",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.window).mean()
        
        return atr / (close + 1e-10)


class VolatilityRatio(BaseFactor):
    """Short-term vs long-term volatility ratio."""
    
    def __init__(self, short: int = 10, long: int = 63):
        self.short = short
        self.long = long
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name="vol_ratio",
            category=FactorCategory.VOLATILITY,
            description="Short/long volatility ratio",
            lookback=self.long
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df['close'].pct_change()
        vol_short = returns.rolling(self.short).std()
        vol_long = returns.rolling(self.long).std()
        return vol_short / (vol_long + 1e-10)


class Skewness(BaseFactor):
    """Return skewness."""
    
    def __init__(self, window: int = 21):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"skewness_{self.window}d",
            category=FactorCategory.VOLATILITY,
            description=f"{self.window}-day return skewness",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df['close'].pct_change()
        return returns.rolling(self.window).skew()


class VolumeZScore(BaseFactor):
    """Volume Z-score."""
    
    def __init__(self, window: int):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"volume_zscore_{self.window}d",
            category=FactorCategory.VOLUME,
            description=f"Volume Z-score ({self.window}d)",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        volume = df['volume']
        mean = volume.rolling(self.window).mean()
        std = volume.rolling(self.window).std()
        return (volume - mean) / (std + 1e-10)


class RelativeVolume(BaseFactor):
    """Volume relative to average."""
    
    def __init__(self, window: int = 21):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name="relative_volume",
            category=FactorCategory.VOLUME,
            description="Relative volume vs 21-day avg",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        volume = df['volume']
        return volume / (volume.rolling(self.window).mean() + 1e-10)


class AmihudIlliquidity(BaseFactor):
    """
    Amihud Illiquidity Ratio.
    Higher = less liquid = potentially higher returns.
    """
    
    def __init__(self, window: int = 21):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"amihud_{self.window}d",
            category=FactorCategory.MICROSTRUCTURE,
            description="Amihud illiquidity",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df['close'].pct_change().abs()
        dollar_vol = df['close'] * df['volume']
        daily_illiq = returns / (dollar_vol + 1e-10)
        log_illiq = np.log1p(daily_illiq * 1e6)  # Log scale
        return log_illiq.rolling(self.window).mean()


class PriceVolumeCorr(BaseFactor):
    """Price-volume correlation."""
    
    def __init__(self, window: int = 21):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"pv_corr_{self.window}d",
            category=FactorCategory.VOLUME,
            description="Price-volume correlation",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df['close'].pct_change()
        vol_change = df['volume'].pct_change()
        return returns.rolling(self.window).corr(vol_change)


def get_microstructure_factors() -> List[BaseFactor]:
    """Get all microstructure/volatility/volume factors."""
    factors = []
    
    # Volatility
    for window in settings.features.volatility_windows:
        factors.append(Volatility(window))
    
    # ATR
    factors.append(ATR(14))
    
    # Volatility ratio
    factors.append(VolatilityRatio(10, 63))
    
    # Skewness
    factors.append(Skewness(21))
    
    # Volume factors
    factors.append(VolumeZScore(10))
    factors.append(VolumeZScore(21))
    factors.append(RelativeVolume(21))
    
    # Microstructure
    factors.append(AmihudIlliquidity(21))
    factors.append(PriceVolumeCorr(21))
    
    return factors