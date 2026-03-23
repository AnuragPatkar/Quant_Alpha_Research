"""
features/fundamental/utils.py
==============================
Shared utilities for all fundamental factor classes.

Purpose
-------
Provides three building blocks used across value.py, quality.py,
growth.py, and financial_health.py:

  FundamentalColumnValidator  — maps logical column names to actual DataFrame
                                column names via COLUMN_MAPPINGS in config/mappings.py.
                                Decouples factor logic from raw data schema.

  SingleColumnFactor          — base class for factors that expose one mapped
                                column directly (e.g. ROE, gross_margin).

  RatioFactor                 — base class for factors computed as num / den
                                where both sides come from mapped columns.

BUG-028 FIX: This entire file was missing from the project. Every fundamental
and financial-health factor imports from '.utils' at module load time, so the
ImportError crashed the full feature pipeline before a single factor computed.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.logging_config import logger
from config.mappings import COLUMN_MAPPINGS
from ..base import FundamentalFactor

EPS = 1e-9


# ---------------------------------------------------------------------------
# FundamentalColumnValidator
# ---------------------------------------------------------------------------

class FundamentalColumnValidator:
    """
    Maps logical column keys (e.g. 'pe_ratio') to actual column names present
    in a DataFrame, using the alias lists defined in config/mappings.py.

    Design
    ------
    - Returns the FIRST alias that exists in the DataFrame, or None.
    - Column lookup is case-insensitive (compares lowercased names).
    - Result is cached per (frozenset(df.columns), key) to avoid repeated scans
      on the same DataFrame within a single factor computation.

    Usage
    -----
        col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        if col:
            pe = df[col]
    """

    # Class-level cache: (frozenset of column names, logical key) -> actual col name
    _cache: dict = {}

    @classmethod
    def find_column(cls, df: pd.DataFrame, key: str) -> Optional[str]:
        """
        Return the first DataFrame column that matches any alias for *key*.

        Parameters
        ----------
        df  : DataFrame to search in
        key : Logical column key from COLUMN_MAPPINGS (e.g. 'pe_ratio')

        Returns
        -------
        str   — actual column name if found
        None  — if no alias matches any column in df
        """
        # Fast path: if the key itself is directly in the DataFrame
        if key in df.columns:
            return key

        # Cache key: use a frozen set of lowercased columns + logical key
        col_set = frozenset(c.lower() for c in df.columns)
        cache_key = (col_set, key)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        # Look up aliases from COLUMN_MAPPINGS
        aliases = COLUMN_MAPPINGS.get(key, [])
        if not aliases:
            # The key is not in COLUMN_MAPPINGS; try direct lookup only
            cls._cache[cache_key] = None
            return None

        # Build a lowercase → actual column name lookup for this DataFrame
        lower_to_actual = {c.lower(): c for c in df.columns}

        result = None
        for alias in aliases:
            match = lower_to_actual.get(alias.lower())
            if match is not None:
                result = match
                break

        cls._cache[cache_key] = result
        return result

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the column-lookup cache (call between DataFrames with different schemas)."""
        cls._cache.clear()

    @classmethod
    def get_aliases(cls, key: str) -> list:
        """Return all aliases registered for *key* (including the key itself)."""
        aliases = COLUMN_MAPPINGS.get(key, [])
        if key not in aliases:
            return [key] + list(aliases)
        return list(aliases)


# ---------------------------------------------------------------------------
# SingleColumnFactor
# ---------------------------------------------------------------------------

class SingleColumnFactor(FundamentalFactor):
    """
    Base class for fundamental factors that expose a single mapped column.

    Subclass contract
    -----------------
    Pass the logical column key and an optional `invert` flag:

        class ROE(SingleColumnFactor):
            def __init__(self):
                super().__init__('qual_roe', 'roe', description='Return on Equity')

        class LowLeverage(SingleColumnFactor):
            def __init__(self):
                super().__init__(
                    'qual_low_leverage', 'debt_equity',
                    invert=True, description='Inverted D/E'
                )

    Parameters
    ----------
    factor_name : str  — output column name (e.g. 'qual_roe')
    col_key     : str  — logical key in COLUMN_MAPPINGS (e.g. 'roe')
    invert      : bool — if True, return -1 × value (so "higher = better" is preserved)
    description : str  — human-readable description
    """

    def __init__(
        self,
        factor_name: str,
        col_key: str,
        invert: bool = False,
        description: str = '',
        **kwargs,
    ):
        super().__init__(name=factor_name, description=description, **kwargs)
        self.col_key = col_key
        self.invert = invert

    def compute(self, df: pd.DataFrame) -> pd.Series:
        col = FundamentalColumnValidator.find_column(df, self.col_key)
        if col is None:
            logger.warning(
                f"⚠️  {self.name}: column '{self.col_key}' not found "
                f"(aliases: {FundamentalColumnValidator.get_aliases(self.col_key)})"
            )
            return pd.Series(np.nan, index=df.index)

        values = df[col].copy()

        if self.invert:
            values = -1.0 * values

        return values


# ---------------------------------------------------------------------------
# RatioFactor
# ---------------------------------------------------------------------------

class RatioFactor(FundamentalFactor):
    """
    Base class for fundamental factors computed as numerator / denominator.

    Subclass contract
    -----------------
    Pass logical column keys for numerator and denominator:

        class FCFYield(RatioFactor):
            def __init__(self):
                super().__init__(
                    'val_fcf_yield',
                    num_key='fcf', den_key='market_cap',
                    description='FCF Yield'
                )

    Parameters
    ----------
    factor_name : str  — output column name
    num_key     : str  — logical numerator key in COLUMN_MAPPINGS
    den_key     : str  — logical denominator key in COLUMN_MAPPINGS
    description : str  — human-readable description
    clip_lower  : float or None — optional lower clip applied to ratio
    clip_upper  : float or None — optional upper clip applied to ratio
    """

    def __init__(
        self,
        factor_name: str,
        num_key: str,
        den_key: str,
        description: str = '',
        clip_lower: Optional[float] = None,
        clip_upper: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(name=factor_name, description=description, **kwargs)
        self.num_key = num_key
        self.den_key = den_key
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

    def compute(self, df: pd.DataFrame) -> pd.Series:
        num_col = FundamentalColumnValidator.find_column(df, self.num_key)
        den_col = FundamentalColumnValidator.find_column(df, self.den_key)

        if num_col is None:
            logger.warning(
                f"⚠️  {self.name}: numerator '{self.num_key}' not found "
                f"(aliases: {FundamentalColumnValidator.get_aliases(self.num_key)})"
            )
            return pd.Series(np.nan, index=df.index)

        if den_col is None:
            logger.warning(
                f"⚠️  {self.name}: denominator '{self.den_key}' not found "
                f"(aliases: {FundamentalColumnValidator.get_aliases(self.den_key)})"
            )
            return pd.Series(np.nan, index=df.index)

        # Replace exact zeros in denominator to avoid Inf; preserve NaN propagation
        denominator = df[den_col].replace(0, np.nan)
        ratio = df[num_col] / (denominator + EPS)

        if self.clip_lower is not None or self.clip_upper is not None:
            ratio = ratio.clip(lower=self.clip_lower, upper=self.clip_upper)

        return ratio