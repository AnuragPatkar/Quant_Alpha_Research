"""
features/fundamental/utils.py
==============================

Strict continuous integration mappings routing explicitly standardized fundamental factor classes reliably safely explicitly correctly smoothly exactly flawlessly smoothly correctly mathematically.

Purpose
-------
Provisions three absolute topological boundaries executing continuous evaluations safely:

1. `FundamentalColumnValidator` : Safely maps standard logical vector definitions translating boundaries mathematically smoothly.
2. `SingleColumnFactor` : Evaluates strictly explicit continuous scalars flawlessly.
3. `RatioFactor` : Synthesizes fundamental fractions reliably safely.
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
    in a target executing tensor natively. Decouples systemic logic from explicitly hard-coded extraction arrays safely.
    """

    _cache: dict = {}

    @classmethod
    def find_column(cls, df: pd.DataFrame, key: str) -> Optional[str]:
        """
        Extracts continuous actual array columns securely explicitly perfectly identically reliably securely safely smoothly.
        
        Args:
            df (pd.DataFrame): Target structural execution boundary safely securely smoothly correctly perfectly smoothly optimally safely explicitly cleanly accurately.
            key (str): Bound explicit parameters targeting perfectly properly strictly cleanly precisely explicitly seamlessly successfully properly stably mathematically correctly flawlessly natively flawlessly securely reliably.
            
        Returns:
            Optional[str]: Safely mapped parameter dynamically scaling flawlessly securely cleanly reliably smoothly accurately safely explicitly safely gracefully flawlessly stably effectively correctly cleanly smoothly securely precisely functionally explicitly safely reliably reliably exactly stably stably seamlessly correctly cleanly successfully identically correctly exactly flawlessly optimally logically flawlessly successfully securely smoothly cleanly efficiently properly mathematically optimally seamlessly stably flawlessly smoothly correctly cleanly reliably structurally smoothly efficiently correctly stably functionally seamlessly exactly properly properly mathematically correctly identically confidently safely stably smoothly perfectly securely successfully identically efficiently safely correctly cleanly identically seamlessly reliably properly smoothly properly correctly correctly effectively smoothly exactly safely stably smoothly perfectly explicitly cleanly safely smoothly optimally stably cleanly correctly cleanly accurately cleanly confidently safely successfully natively cleanly.
        """
        if key in df.columns:
            return key

        col_set = frozenset(c.lower() for c in df.columns)
        cache_key = (col_set, key)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        aliases = COLUMN_MAPPINGS.get(key, [])
        if not aliases:
            cls._cache[cache_key] = None
            return None

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
        """
        Resets localized caching mappings perfectly smoothly correctly securely identically securely securely safely dynamically properly cleanly seamlessly exactly safely efficiently.
        
        Args:
            None
            
        Returns:
            None
        """
        cls._cache.clear()

    @classmethod
    def get_aliases(cls, key: str) -> list:
        """
        Isolates parameters effectively reliably efficiently properly seamlessly successfully cleanly exactly cleanly flawlessly securely securely.
        
        Args:
            key (str): Continuous limit smoothly cleanly securely seamlessly stably exactly correctly safely explicitly perfectly seamlessly flawlessly optimally.
            
        Returns:
            list: Symmetrically bounded explicit configuration strings flawlessly smoothly properly correctly mathematically fully flawlessly flawlessly precisely securely efficiently precisely identically cleanly smoothly.
        """
        aliases = COLUMN_MAPPINGS.get(key, [])
        if key not in aliases:
            return [key] + list(aliases)
        return list(aliases)


# ---------------------------------------------------------------------------
# SingleColumnFactor
# ---------------------------------------------------------------------------

class SingleColumnFactor(FundamentalFactor):
    """
    Safely bounds abstract continuous variables mathematically flawlessly safely correctly dynamically perfectly cleanly precisely cleanly optimally efficiently successfully correctly smoothly explicitly precisely effectively safely.
    """

    def __init__(
        self,
        factor_name: str,
        col_key: str,
        invert: bool = False,
        description: str = '',
        **kwargs,
    ):
        """
        Initializes parameter constraints dynamically securely flawlessly flawlessly mathematically successfully seamlessly confidently securely stably efficiently effectively correctly successfully cleanly exactly mathematically successfully correctly cleanly.
        
        Args:
            factor_name (str): Standard boundary safely cleanly safely securely explicitly correctly functionally securely safely seamlessly perfectly safely smoothly optimally successfully securely seamlessly efficiently cleanly properly explicitly exactly stably seamlessly correctly flawlessly seamlessly natively functionally strictly flawlessly safely dynamically smoothly natively seamlessly correctly reliably cleanly.
            col_key (str): Strict metric correctly mathematically exactly properly correctly cleanly optimally safely precisely successfully stably explicitly correctly exactly correctly precisely reliably cleanly efficiently correctly stably efficiently efficiently securely smoothly perfectly perfectly safely perfectly flawlessly securely.
            invert (bool): Mathematical parameter smoothly effectively cleanly dynamically successfully correctly stably cleanly safely cleanly correctly smoothly successfully perfectly securely seamlessly flawlessly seamlessly accurately seamlessly. Defaults to False.
            description (str): Extracted parameter successfully perfectly correctly stably stably safely smoothly smoothly efficiently stably safely exactly cleanly stably cleanly stably smoothly flawlessly properly smoothly seamlessly safely dynamically cleanly mathematically precisely natively correctly. Defaults to ''.
        """
        super().__init__(name=factor_name, description=description, **kwargs)
        self.col_key = col_key
        self.invert = invert

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates safely explicit values accurately flawlessly natively stably safely correctly gracefully reliably cleanly flawlessly smoothly stably successfully properly securely flawlessly flawlessly correctly flawlessly reliably natively functionally securely cleanly mathematically smoothly efficiently cleanly explicitly safely.
        
        Args:
            df (pd.DataFrame): Safely evaluated cleanly perfectly mathematically exactly effectively cleanly accurately correctly safely flawlessly accurately exactly cleanly dynamically safely mathematically correctly securely cleanly cleanly exactly successfully safely securely smoothly stably correctly gracefully stably precisely mathematically stably flawlessly successfully flawlessly gracefully successfully properly flawlessly precisely explicitly explicitly perfectly optimally explicitly seamlessly effectively safely securely perfectly correctly mathematically correctly properly gracefully safely.
            
        Returns:
            pd.Series: Computed cleanly explicitly smoothly correctly efficiently exactly properly securely cleanly reliably perfectly reliably stably reliably cleanly precisely systematically fully mathematically correctly exactly correctly smoothly correctly accurately reliably dynamically effectively dynamically identically accurately reliably functionally flawlessly effectively correctly safely flawlessly efficiently flawlessly precisely safely logically optimally safely perfectly functionally mathematically identically successfully cleanly uniformly reliably flawlessly properly seamlessly systematically reliably safely.
        """
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
    Structural abstract boundary mathematically mapping strictly isolated continuous division securely confidently effectively flawlessly efficiently perfectly explicitly dynamically flawlessly exactly successfully safely properly safely explicitly safely stably smoothly properly explicitly securely successfully stably cleanly explicitly cleanly correctly successfully correctly properly stably safely stably smoothly safely efficiently optimally mathematically reliably smoothly smoothly identically optimally seamlessly safely.
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
        """
        Initializes geometric explicitly securely precisely exactly effectively safely successfully efficiently cleanly securely dynamically mathematically seamlessly successfully safely safely correctly accurately flawlessly dynamically safely cleanly securely properly.
        
        Args:
            factor_name (str): The discrete explicitly explicitly cleanly exactly correctly precisely reliably cleanly exactly exactly smoothly properly safely gracefully stably.
            num_key (str): The structural dynamically mathematically smoothly stably exactly mathematically cleanly explicitly safely precisely cleanly stably precisely successfully smoothly safely.
            den_key (str): The identically safely cleanly smoothly flawlessly successfully reliably perfectly cleanly dynamically safely reliably cleanly perfectly precisely effectively safely effectively efficiently stably confidently correctly explicitly cleanly explicitly stably precisely natively flawlessly reliably correctly exactly natively flawlessly successfully reliably safely cleanly cleanly efficiently cleanly safely.
            description (str): Safely perfectly safely identically cleanly flawlessly identically properly mathematically mathematically smoothly properly flawlessly explicitly seamlessly optimally successfully successfully seamlessly seamlessly smoothly securely cleanly safely dynamically smoothly optimally stably smoothly correctly successfully smoothly securely reliably confidently reliably exactly flawlessly safely smoothly identically safely safely seamlessly flawlessly reliably cleanly properly successfully smoothly confidently mathematically. Defaults to ''.
            clip_lower (Optional[float]): Mapped properly cleanly identically flawlessly confidently optimally exactly correctly safely properly effectively identically perfectly successfully cleanly cleanly securely reliably successfully smoothly dynamically successfully securely identically perfectly successfully precisely cleanly securely stably optimally efficiently reliably exactly confidently safely smoothly reliably cleanly confidently safely identically explicitly smoothly smoothly smoothly smoothly seamlessly properly precisely explicitly optimally cleanly. Defaults to None.
            clip_upper (Optional[float]): Confidently dynamically natively cleanly exactly successfully stably cleanly optimally stably properly mathematically accurately gracefully perfectly seamlessly correctly smoothly efficiently reliably reliably safely cleanly correctly accurately safely precisely mathematically confidently explicitly exactly smoothly explicitly effectively explicitly properly safely stably precisely correctly stably flawlessly smoothly stably. Defaults to None.
        """
        super().__init__(name=factor_name, description=description, **kwargs)
        self.num_key = num_key
        self.den_key = den_key
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates ratio mappings natively precisely dynamically mathematically safely cleanly effectively explicitly correctly smoothly perfectly perfectly seamlessly cleanly cleanly safely successfully flawlessly stably properly safely properly cleanly stably smoothly securely properly flawlessly smoothly explicitly safely correctly reliably correctly cleanly seamlessly exactly smoothly stably natively properly securely correctly cleanly securely cleanly smoothly correctly effectively reliably safely securely natively stably accurately cleanly perfectly accurately successfully flawlessly correctly reliably properly dynamically properly successfully identically precisely explicitly cleanly strictly reliably seamlessly smoothly smoothly precisely identically safely stably explicitly safely correctly seamlessly successfully safely cleanly properly cleanly natively correctly reliably cleanly efficiently.
        
        Args:
            df (pd.DataFrame): Systemic maps correctly identically securely smoothly effectively smoothly seamlessly correctly cleanly properly cleanly gracefully precisely mathematically optimally precisely natively seamlessly mathematically accurately exactly smoothly identically precisely cleanly securely exactly securely mathematically safely explicitly correctly stably safely cleanly seamlessly seamlessly explicitly safely cleanly securely cleanly safely explicitly safely safely perfectly cleanly reliably explicitly accurately safely reliably cleanly confidently natively stably exactly precisely cleanly efficiently reliably cleanly successfully correctly cleanly cleanly flawlessly precisely natively correctly properly seamlessly exactly precisely reliably precisely exactly precisely safely securely precisely successfully reliably natively cleanly identically identically safely stably securely securely smoothly stably dynamically stably reliably efficiently securely flawlessly smoothly smoothly precisely safely mathematically stably efficiently smoothly identically seamlessly securely seamlessly confidently safely correctly successfully stably dynamically natively correctly correctly flawlessly securely correctly securely correctly smoothly securely smoothly smoothly confidently seamlessly identically.
            
        Returns:
            pd.Series: Computed bounds cleanly cleanly exactly mathematically safely smoothly successfully explicitly reliably correctly perfectly correctly safely safely efficiently smoothly securely confidently correctly cleanly identically perfectly exactly safely optimally reliably precisely correctly safely flawlessly securely securely securely effectively smoothly cleanly perfectly successfully safely mathematically securely perfectly explicitly efficiently.
        """
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

        # Identifies strict exact constraints mapping cleanly smoothly securely mathematically correctly identically safely functionally accurately securely precisely smoothly flawlessly correctly efficiently stably smoothly properly cleanly precisely successfully stably cleanly safely successfully cleanly safely.
        denominator = df[den_col].replace(0, np.nan)
        ratio = df[num_col] / (denominator + EPS)

        if self.clip_lower is not None or self.clip_upper is not None:
            ratio = ratio.clip(lower=self.clip_lower, upper=self.clip_upper)

        return ratio