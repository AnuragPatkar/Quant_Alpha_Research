"""
Momentum Factors
================
Trend-following alpha factors.
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


class ReturnMomentum(BaseFactor):
    """N-day price momentum (return)."""
    
    def __init__(self, window: int):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"mom_{self.window}d",
            category=FactorCategory.MOMENTUM,
            description=f"{self.window}-day return momentum",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df['close'].pct_change(self.window)


class RateOfChange(BaseFactor):
    """Rate of change indicator."""
    
    def __init__(self, window: int = 21):
        self.window = window
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name=f"roc_{self.window}d",
            category=FactorCategory.MOMENTUM,
            description=f"{self.window}-day ROC",
            lookback=self.window
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        return (close - close.shift(self.window)) / (close.shift(self.window) + 1e-10)


class EMAMomentum(BaseFactor):
    """EMA-based momentum (MACD-like)."""
    
    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow
    
    @property
    def info(self) -> FactorInfo:
        return FactorInfo(
            name="ema_momentum",
            category=FactorCategory.MOMENTUM,
            description="EMA crossover momentum",
            lookback=self.slow
        )
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        return (ema_fast - ema_slow) / (close + 1e-10)


def get_momentum_factors() -> List[BaseFactor]:
    """Get all momentum factors."""
    factors = []
    
    # Return momentum at different horizons
    for window in settings.features.momentum_windows:
        factors.append(ReturnMomentum(window))
    
    # ROC
    factors.append(RateOfChange(21))
    
    # EMA momentum
    factors.append(EMAMomentum(12, 26))
    
    return factors