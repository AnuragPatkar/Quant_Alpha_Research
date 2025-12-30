"""
Base Factor Classes
===================
Abstract base class for all alpha factors.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass
from enum import Enum


class FactorCategory(Enum):
    """Factor categories."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    MICROSTRUCTURE = "microstructure"


@dataclass
class FactorInfo:
    """Factor metadata."""
    name: str
    category: FactorCategory
    description: str
    lookback: int


class BaseFactor(ABC):
    """
    Abstract base class for factors.
    
    All factors must implement:
    - info: Factor metadata
    - compute(): Calculate factor values
    """
    
    @property
    @abstractmethod
    def info(self) -> FactorInfo:
        """Return factor metadata."""
        pass
    
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute factor values.
        
        Args:
            df: OHLCV DataFrame for single stock
            
        Returns:
            Series of factor values
        """
        pass