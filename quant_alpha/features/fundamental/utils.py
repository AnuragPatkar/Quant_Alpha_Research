import numpy as np
import pandas as pd
from typing import Optional
from config.logging_config import logger
from config.mappings import COLUMN_MAPPINGS
from ..base import FundamentalFactor, EPS

class FundamentalColumnValidator:
    """
    Centralized Knowledge Base for Column Mapping.
    Matches 'fundamentals.parquet' schema.
    """
    @classmethod
    def find_column(cls, df: pd.DataFrame, key: str) -> Optional[str]:
        # 1. Direct Check
        if key in df.columns: return key
        # 2. Mapping Check
        if key in COLUMN_MAPPINGS:
            for variant in COLUMN_MAPPINGS[key]:
                if variant in df.columns: return variant
        # 3. Case-Insensitive
        col_map_lower = {c.lower(): c for c in df.columns}
        if key.lower() in col_map_lower: return col_map_lower[key.lower()]
        return None

# ==================== SHARED FACTOR BASES ====================

class SingleColumnFactor(FundamentalFactor):
    """
    Parent class for factors that just retrieve 1 column.
    Handles lookup, optional inversion (for Risk metrics), and logging.
    """
    def __init__(self, name: str, col_key: str, invert: bool = False, description: str = ""):
        super().__init__(name=name, description=description)
        self.col_key = col_key
        self.invert = invert

    def compute(self, df: pd.DataFrame) -> pd.Series:
        col = FundamentalColumnValidator.find_column(df, self.col_key)
        if col:
            val = df[col]
            return -1.0 * val if self.invert else val
        
        logger.warning(f"⚠️  {self.name}: Missing '{self.col_key}' column")
        return pd.Series(np.nan, index=df.index)

class RatioFactor(FundamentalFactor):
    """
    Parent class for factors calculated as A / B.
    """
    def __init__(self, name: str, num_key: str, den_key: str, description: str = ""):
        super().__init__(name=name, description=description)
        self.num_key = num_key
        self.den_key = den_key

    def compute(self, df: pd.DataFrame) -> pd.Series:
        num = FundamentalColumnValidator.find_column(df, self.num_key)
        den = FundamentalColumnValidator.find_column(df, self.den_key)
        
        if num and den:
            # Handle division by zero safely using EPS
            return df[num] / (df[den] + EPS)
            
        logger.warning(f"⚠️  {self.name}: Missing {self.num_key} or {self.den_key}")
        return pd.Series(np.nan, index=df.index)