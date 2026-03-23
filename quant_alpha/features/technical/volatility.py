"""
Volatility Factors
==================
Quantitative signals capturing market risk, variance dynamics, and tail events.

Purpose
-------
This module constructs alpha factors quantifying the dispersion of asset returns.
It implements estimators ranging from simple Close-to-Close volatility to
path-dependent metrics like Garman-Klass and ATR. These factors serve as critical
inputs for:
1.  **Risk Management**: Position sizing (Vol-Targeting) and stop-loss calibration.
2.  **Regime Detection**: Identifying transitions between mean-reverting (low vol)
    and trending (high vol) states.
3.  **Tail Risk Hedging**: Skewness and Kurtosis signals for crash anticipation.

Usage
-----
Factors are automatically registered with the `FactorRegistry` upon import.

.. code-block:: python

    registry = FactorRegistry()
    vol_factor = registry.get('volatility_21d')
    signals = vol_factor.compute(market_data_df)

Importance
----------
- **Predictive Power**: Volatility clustering (GARCH effects) makes volatility
  highly autocorrelated and predictable, unlike raw returns.
- **Risk Parity**: Essential for normalizing alpha signals across assets with
  heterogeneous risk profiles (e.g., Tech vs. Utilities).
- **Efficiency**: Garman-Klass estimators provide statistical efficiency gains over
  standard estimators by incorporating intraday (OHLC) excursions.

Tools & Frameworks
------------------
- **Pandas**: Efficient `rolling` window operations and `groupby` transformations.
- **NumPy**: Vectorized logarithmic and square root calculations for estimator derivation.
- **FactorRegistry**: Decorator-based registration for pipeline integration.
"""

import numpy as np
import pandas as pd
from ..registry import FactorRegistry
from ..base import TechnicalFactor
from quant_alpha.utils.column_helpers import safe_col

EPS = 1e-9  # Machine epsilon to prevent DivisionByZero

# ==================== 1. STANDARD HISTORICAL VOLATILITY ====================
# Metric: Annualized Rolling Standard Deviation.
# Formula: $$ \sigma_{ann} = \text{std}(R_t, N) \times \sqrt{252} $$

# REMOVED: Volatility5D (too short-term noise)
# @FactorRegistry.register()
# class Volatility5D(TechnicalFactor):
#     def __init__(self, period=5):
#         super().__init__(name='volatility_5d', description='1 Week volatility', lookback_period=period + 1)
#         self.period = period
#     
#     def compute(self, df: pd.DataFrame) -> pd.Series:
#         return df.groupby('ticker')['close'].transform(lambda x: x.pct_change().rolling(window=self.period).std() * np.sqrt(252))

# REMOVED: Volatility10D (redundant with Volatility21D)
# @FactorRegistry.register()
# class Volatility10D(TechnicalFactor):
#     def __init__(self, period=10):
#         super().__init__(name='volatility_10d',description='2 Week volatility',lookback_period=period + 1)
#         self.period = period
#     
#     def compute(self, df: pd.DataFrame) -> pd.Series:       
#         return df.groupby('ticker')['close'].transform(lambda x: x.pct_change().rolling(window=self.period).std() * np.sqrt(252))

@FactorRegistry.register()
class Volatility21D(TechnicalFactor):
    """
    1-Month Historical Volatility (21 Trading Days).
    Baseline metric for risk normalization.
    """
    def __init__(self, period=21):
        super().__init__(name='volatility_21d', description='1 Month Volatility', lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Optimization: Transform preserves the original index structure ($O(N)$).
        return df.groupby('ticker')['close'].transform(
            lambda x: x.pct_change().rolling(window=self.period).std() * np.sqrt(252)
        )
    
@FactorRegistry.register()
class Volatility63D(TechnicalFactor):
    """
    3-Month Historical Volatility (Quarterly).
    Used for longer-term regime identification.
    """
    def __init__(self, period=63):
        super().__init__(name='volatility_63d', description='3 Month Volatility', lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].transform(
            lambda x: x.pct_change().rolling(window=self.period).std() * np.sqrt(252)
        )

# REMOVED: Volatility126D (less useful, regime-like)
# @FactorRegistry.register()
# class Volatility126D(TechnicalFactor):
#     def __init__(self, period=126):
#         super().__init__(name='volatility_126d', description='6 Month Volatility', lookback_period=period + 1)
#         self.period = period
#     def compute(self, df: pd.DataFrame) -> pd.Series:
#         return df.groupby('ticker')['close'].transform(
#             lambda x: x.pct_change().rolling(window=self.period).std() * np.sqrt(252)
#         )
    
# ==================== 2. GARMAN-KLASS VOLATILITY ====================
# Metric: Range-based Volatility Estimator.
# Formula:
# $$ \sigma^2_{GK} = 0.5 \ln\left(\frac{H_t}{L_t}\right)^2 - (2\ln(2)-1)\ln\left(\frac{C_t}{O_t}\right)^2 $$

@FactorRegistry.register()
class GKVolatility21D(TechnicalFactor):
    """
    Garman-Klass Volatility (21D).
    
    Captures intraday excursions (High/Low) and overnight gaps (Open/Close).
    More statistically efficient than Close-to-Close volatility.
    """
    def __init__(self, period=21):
        super().__init__(name='gk_vol_21',description='Garman-Klass Volatility',lookback_period=period )
        self.period = period

    def compute(self, df:pd.DataFrame) -> pd.Series :
        high = safe_col(df, "high")
        low = safe_col(df, "low")
        open_ = safe_col(df, "open")
        close = safe_col(df, "close")
        
        if high.isna().all():
            return pd.Series(np.nan, index=df.index)

        # Pre-calculate logarithmic returns (Vectorized $O(N)$)
        log_hl = np.log((high / low).replace(0,np.nan))
        log_co = np.log((close / open_).replace(0,np.nan))
        
        # Variance Estimator per period
        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)

        # Rolling Variance Mean -> StdDev -> Annualize
        # Note: Using transform ensures strict index alignment with the input DataFrame
        return gk_var.groupby(df['ticker']).transform(lambda x: np.sqrt(x.rolling(window=self.period).mean().clip(lower=0)) * np.sqrt(252))
    
@FactorRegistry.register()
class GKVolatility63D(TechnicalFactor):
    """Garman-Klass Volatility (63D)."""
    def __init__(self, period = 63):
        super().__init__(name='gk_val_63',description='Garman-Klass Volatility (63D)',lookback_period=period )
        self.period = period

    def compute(self, df:pd.DataFrame) -> pd.Series :
        high = safe_col(df, "high")
        low = safe_col(df, "low")
        open_ = safe_col(df, "open")
        close = safe_col(df, "close")
        
        if high.isna().all():
            return pd.Series(np.nan, index=df.index)

        log_hl = np.log((high / low).replace(0, np.nan))
        log_co = np.log((close / open_).replace(0, np.nan))
        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        
        return gk_var.groupby(df['ticker']).transform(
            lambda x: np.sqrt(x.rolling(window=self.period).mean().clip(lower=0)) * np.sqrt(252)
        )
    
# ==================== 3. ATR (AVERAGE TRUE RANGE) ====================
# Metric: Normalized Average True Range.
# Purpose: Measures absolute price movement normalized by price level.

@FactorRegistry.register()
class ATR14(TechnicalFactor):
    """
    ATR 14 (Normalized).
    
    Formula: $$ \text{NATR} = \frac{\text{EMA}(\text{TrueRange}, 14)}{\text{Close}} $$
    """
    def __init__(self, period=14):
        super().__init__(name='atr_14', description='ATR 14 Normalized', lookback_period=period +1 )
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_atr(group):
            high = safe_col(group, "high")
            low = safe_col(group, "low")
            if high.isna().all():
                return pd.Series(np.nan, index=group.index)
            prev_close = group['close'].shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window=self.period).mean()

        atr = df.groupby('ticker', group_keys=False).apply(calc_atr, include_groups=False)
        return atr / (df['close'] + EPS)
    
@FactorRegistry.register()
class ATR21(TechnicalFactor):
    """ATR 21 (Normalized)."""
    def __init__(self, period=21):
        super().__init__(name='atr_21', description='ATR 21 Normalized', lookback_period=period + 1)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_atr(group):
            high = safe_col(group, "high")
            low = safe_col(group, "low")
            if high.isna().all():
                return pd.Series(np.nan, index=group.index)
            prev_close = group['close'].shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window=self.period).mean()

        atr = df.groupby('ticker', group_keys=False).apply(calc_atr, include_groups=False)
        return atr / (df['close'] + EPS)
    
# ==================== 4. REGIME INDICATORS (RATIO, SKEW, KURT) ====================

@FactorRegistry.register()
class VolatilityRatio(TechnicalFactor):
    """
    Volatility Regime Ratio (5D / 21D).
    
    Formula: $$ VR = \frac{\sigma_{5d}}{\sigma_{21d}} $$
    
    Signals:
    - $VR > 1.0$: Volatility expansion (Shock).
    - $VR < 1.0$: Volatility contraction (Stabilization).
    """
    def __init__(self, short_period=5, long_period=21):
        super().__init__(name='vol_ratio_5_21', description='Vol Ratio 5D/21D', lookback_period=long_period + 1)
        self.short = short_period
        self.long = long_period

    def compute(self, df:pd.DataFrame) -> pd.Series:
        # Use transform instead of rolling().std() directly to maintain Index alignment
        vol_short = df.groupby('ticker')['close'].transform(lambda x:x.pct_change().rolling(window=self.short).std())
        vol_long = df.groupby('ticker')['close'].transform(lambda x:x.pct_change().rolling(window=self.long).std())
        return vol_short / (vol_long + EPS)
    
@FactorRegistry.register()
class Skewness21D(TechnicalFactor):
    """
    Rolling Skewness (21D).
    
    Measures the asymmetry of the return distribution.
    - Negative Skew: Higher probability of large losses (Tail Risk).
    - Positive Skew: Higher probability of large gains (Lottery-like).
    """
    def __init__(self, period=21):
        super().__init__(name='skew_21d', description='Rolling Skewness 21D', lookback_period=period + 1)
        self.period = period

    def compute(self, df:pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].transform(lambda x: x.pct_change().rolling(window=self.period).skew()).fillna(0)

@FactorRegistry.register()
class Kurtosis21D(TechnicalFactor):
    """
    Rolling Kurtosis (21D).
    
    Measures the "tailedness" of the return distribution (Leptokurtic risk).
    High Kurtosis indicates frequent extreme price moves (Fat Tails).
    """
    def __init__(self, period=21):
        super().__init__(name='kurt_21d', description='Rolling Kurtosis 21D', lookback_period=period + 1)
        self.period = period
    
    def compute(self, df:pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].transform(lambda x:x.pct_change().rolling(window=self.period).kurt()).fillna(0)

    
    
    
