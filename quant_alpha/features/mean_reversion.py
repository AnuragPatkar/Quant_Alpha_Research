"""
Mean Reversion Factors
======================
Counter-trend alpha factors.
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


class RSI(BaseFactor):
    """Relative Strength Index."""
    
    def __init__(self, window: int = 14):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"rsi_{self.window}d",
            category=FactorCategory.MEAN_REVERSION,
            description=f"{self.window}-day RSI",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.window).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))


class DistanceFromMA(BaseFactor):
    """Distance from moving average."""
    
    def __init__(self, window: int):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"dist_ma_{self.window}d",
            category=FactorCategory.MEAN_REVERSION,
            description=f"Distance from {self.window}-day MA",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        ma = close.rolling(self.window).mean()
        return (close - ma) / (ma + 1e-10)


class ZScore(BaseFactor):
    """Price Z-score."""
    
    def __init__(self, window: int = 21):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"zscore_{self.window}d",
            category=FactorCategory.MEAN_REVERSION,
            description=f"{self.window}-day price Z-score",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        mean = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        return (close - mean) / (std + 1e-10)


class BollingerPosition(BaseFactor):
    """Position within Bollinger Bands (0 to 1)."""
    
    def __init__(self, window: int = 20):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name="bb_position",
            category=FactorCategory.MEAN_REVERSION,
            description="Bollinger Band position (0-1)",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return (close - lower) / (upper - lower + 1e-10)


def get_mean_reversion_factors() -> List[BaseFactor]:
    """Get all mean reversion factors."""
    factors = []
    
    # RSI
    for window in settings.features.rsi_windows:
        factors.append(RSI(window))
    
    # Distance from MA
    for window in settings.features.ma_windows:
        factors.append(DistanceFromMA(window))
    
    # Z-score
    factors.append(ZScore(10))
    factors.append(ZScore(21))
    
    # Bollinger position
    factors.append(BollingerPosition(20))
    
    return factors