"""
Earnings Surprise Factors
=========================
Quantitative signals capturing the market's reaction to earnings announcements relative to expectations.

Purpose
-------
This module generates alpha factors based on "Earnings Surprises"—the divergence between
reported EPS and analyst consensus. It transforms sparse, event-driven earnings data
into continuous daily signals using forward-filling logic. Key metrics include
Standardized Unexpected Earnings (SUE) and surprise momentum, which are primary
drivers of the Post-Earnings Announcement Drift (PEAD) anomaly.

Usage
-----
Factors are registered with the `FactorRegistry` and computed over a standardized
OHLCV + Earnings DataFrame.

.. code-block:: python

    from quant_alpha.features.registry import FactorRegistry
    
    registry = FactorRegistry()
    sue_factor = registry.get('eps_sue_price')
    signals = sue_factor.compute(market_data_df)

Importance
----------
- **Alpha Generation**: Captures PEAD, where prices tend to drift in the direction
  of the earnings surprise for weeks or months following the announcement.
- **Signal Normalization**: Uses price-based standardization ($SUE_p$) to ensure
  comparability across assets with different nominal price levels.
- **Regime Identification**: Consecutive beats (streaks) often indicate a
  fundamental structural shift not yet fully priced in by the consensus.

Tools & Frameworks
------------------
- **Pandas**: Efficient `groupby-apply` patterns for handling ticker-specific event windows.
- **NumPy**: Vectorized arithmetic for calculating percentage deviations and normalization.
- **FactorRegistry**: Decorator-based registration for pipeline integration.
"""

import pandas as pd
import numpy as np  
from ..base import EarningsFactor
from ..registry import FactorRegistry
from config.logging_config import logger
from .utils import detect_earnings_events, get_events_with_surprise

@FactorRegistry.register()
class EPSSurprise(EarningsFactor):
    """
    Standardized EPS Surprise (SUE) normalized by Price.
    
    A robust variation of the classic SUE metric. Normalizing by price instead of
    earnings volatility allows for better cross-sectional comparison between
    high-priced and low-priced stocks.
    
    Formula:
    $$ SUE_{price} = \frac{EPS_{actual} - EPS_{estimate}}{Price_{close}} $$
    """
    def __init__(self):
        """Initializes continuous cross-sectional standardized unexpected earnings bounds."""
        super().__init__(name='eps_sue_price', description='Surprise standardized by Price')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates discrete event boundaries mapping standardized unexpected earnings continuously.
        
        Args:
            df (pd.DataFrame): Systemic temporal map arrays securely bounding event structures.
            
        Returns:
            pd.Series: Evaluated parameter bounds representing continuous point-in-time state.
        """
        if 'eps_actual' not in df.columns or 'eps_estimate' not in df.columns or 'close' not in df.columns:
            return pd.Series(np.nan, index=df.index)
            
        def _calc_sue(group):
            # Extracts exact independent coordinate indices mapping discrete earnings releases
            is_new_event = detect_earnings_events(group)
            events = group.loc[is_new_event].copy()
            
            if events.empty:
                return pd.Series(np.nan, index=group.index)
                
            # Strictly aligns point-in-time market vectors guaranteeing availability
            # for accounting reports systematically released during macro non-trading bounds.
            filled_close = group['close'].ffill()
            events_close = filled_close.loc[events.index]
            valid_close = events_close.where(events_close > 0, np.nan)
            events['sue'] = (events['eps_actual'] - events['eps_estimate']) / valid_close
            
            # Mathematically bounds signal anomalies suppressing execution microstructure artifacts.
            return events['sue'].reindex(group.index).ffill().clip(-0.5, 0.5)
        
        return df.groupby('ticker', group_keys=False).apply(_calc_sue)


@FactorRegistry.register()
class EPSSurprisePercentage(EarningsFactor):
    """
    Raw EPS Surprise Percentage.
    
    Formula:
    $$ Surprise\% = \frac{EPS_{actual} - EPS_{estimate}}{|EPS_{estimate}|} $$
    """
    def __init__(self):
        """Initializes absolute percentage scalar boundaries mapping earnings deviations."""
        super().__init__(name='earn_surprise_pct', description='EPS Surprise Percentage')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates strict independent variations bounding normalized deviation sequences.
        
        Args:
            df (pd.DataFrame): Bounding evaluation parameters safely mapped sequentially.
            
        Returns:
            pd.Series: Symmetrically bounded parameters structurally standardizing event constraints.
        """
        # Identifies structural boundaries securely prioritizing native recalculation 
        # explicitly guaranteeing synchronous temporal alignment.
        cols_needed = ['eps_actual', 'eps_estimate']
        if not all(c in df.columns for c in cols_needed) and 'surprise_pct' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_pct(group):
            events = get_events_with_surprise(group)
            if events.empty: return pd.Series(np.nan, index=group.index)
            
            # Safely clips variance at +/- 500% avoiding extreme denominator collapse artifacts.
            return events['surprise_pct'].reindex(group.index).ffill().clip(-500, 500)

        return df.groupby('ticker', group_keys=False).apply(_calc_pct)


@FactorRegistry.register()
class ConsecutiveSurprise(EarningsFactor):
    """
    Earnings Streak: Count of contiguous quarters with positive surprises.
    
    Identifies companies with a sustained pattern of outperformance, often
    indicative of conservative guidance or superior execution.
    """
    def __init__(self):
        """Initializes continuous run-length execution counting metrics exactly."""
        super().__init__(name='earn_streak', description='Consecutive Positive Surprises')
        
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes aggregate streak persistence applying vectorized state tracking efficiently.
        
        Args:
            df (pd.DataFrame): Explicitly evaluating array parameter definitions exactly mapped.
            
        Returns:
            pd.Series: Continuous parameters isolating historical trajectory streaks explicitly.
        """
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_streak(group):
            events = get_events_with_surprise(group)
            if events.empty:
                return pd.Series(0, index=group.index)

            condition = events['surprise_pct'] > 0
            
            # Vectorized Run-Length Encoding systematically extracting continuous states:
            # Identifies exact indices where evaluation boundaries change synchronously.
            run_ids = (condition != condition.shift()).cumsum()
            events['streak'] = condition.groupby(run_ids).cumsum()
            events['streak'] = events['streak'].where(condition, 0)
            
            return events['streak'].reindex(group.index).ffill().fillna(0)
        
        return df.groupby('ticker', group_keys=False).apply(_calc_streak)
    

@FactorRegistry.register()
class LastQuarterMagnitude(EarningsFactor):
    """
    Absolute Magnitude of Last Quarter's Surprise.
    
    Formula:
    $$ Magnitude_t = |Surprise\%_t| $$
    """
    def __init__(self):
        """Initializes absolute surprise scalar extractions seamlessly."""
        super().__init__(name='earn_last_quarter_magnitude', description='Last Quarter Surprise Magnitude')
        
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates discrete boundaries cleanly matching trailing report magnitudes accurately.
        
        Args:
            df (pd.DataFrame): Foundational execution bounds logically isolated smoothly.
            
        Returns:
            pd.Series: Flawlessly tracked deviation amplitudes bounded strictly to historical states.
        """
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_last_quarter(group):
            events = get_events_with_surprise(group)
            if events.empty:
                return pd.Series(np.nan, index=group.index)
            
            # Evaluates pure magnitude boundaries disregarding specific structural directions.
            events['magnitude'] = events['surprise_pct'].abs()
            events['last_quarter_mag'] = events['magnitude']
            
            return events['last_quarter_mag'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_last_quarter)


@FactorRegistry.register()
class BeatMissMomentum(EarningsFactor):
    """
    Beat/Miss Momentum: Rolling win rate over the last year.
    
    Formula:
    $$ Momentum_t = \frac{1}{4} \sum_{i=0}^{3} \mathbb{I}(Surprise_{t-i} > 0) \times 100 $$
    """
    def __init__(self):
        """Initializes dynamic win-rate probabilistic mappings securely."""
        super().__init__(name='earn_beat_miss_momentum', description='% of Last 4Q Beaten')
        
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates strict point-in-time statistical boundaries representing historical beat probability.
        
        Args:
            df (pd.DataFrame): Execution matrices safely tracking underlying quarterly sequences.
            
        Returns:
            pd.Series: Continuous historical momentum probability bounded symmetrically.
        """
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_momentum(group):
            events = get_events_with_surprise(group)
            if events.empty:
                return pd.Series(np.nan, index=group.index)
            
            # Properly propagates missing evaluations bounding empirical limits natively.
            events['beat'] = np.where(events['surprise_pct'].isna(), np.nan, (events['surprise_pct'] > 0).astype(float))
            
            # Enforces statistical stability boundaries ($N \ge 2$) limiting early-state variance.
            events['momentum'] = events['beat'].rolling(window=4, min_periods=2).mean() * 100
            
            return events['momentum'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_momentum)