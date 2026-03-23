"""
Volume & Liquidity Factors
==========================
Quantitative signals capturing trading activity, liquidity constraints, and institutional flow.

FIXES:
  BUG-041: AccumulationDistribution.compute() — ad_line was built with
           pd.Series(mfv).cumsum() which creates a fresh 0-indexed Series,
           losing the group's original index. The subsequent division and return
           could then silently misalign rows when apply() reassembled the result.
           Fixed by constructing pd.Series(mfv, index=group.index) to preserve
           the original row labels throughout the calculation.
"""

import numpy as np
import pandas as pd
from ..registry import FactorRegistry
from ..base import TechnicalFactor
from quant_alpha.utils.column_helpers import safe_col

EPS = 1e-9


# ==================== 1. VOLUME ANOMALIES (Z-SCORE) ====================

@FactorRegistry.register()
class VolumeZScore21D(TechnicalFactor):
    """
    Volume Z-Score (21D).
    Formula: Z = (V - μ_21) / σ_21
    """
    def __init__(self, period=21):
        super().__init__(
            name='vol_zscore_21d',
            description='Volume Z-Score 21D',
            lookback_period=period + 5
        )
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_z(x):
            return (x - x.rolling(window=self.period).mean()) / (
                x.rolling(window=self.period).std() + EPS
            )
        return df.groupby('ticker')['volume'].transform(calc_z)


# ==================== 2. RELATIVE VOLUME ====================

@FactorRegistry.register()
class VolumeMA20Ratio(TechnicalFactor):
    """
    Relative Volume Ratio (20D).
    Formula: V / SMA(V, 20)
    """
    def __init__(self, period=20):
        super().__init__(
            name='vol_ma20_ratio',
            description='Volume MA20 Ratio',
            lookback_period=period + 5
        )
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma_vol = df.groupby('ticker')['volume'].transform(
            lambda x: x.rolling(window=self.period).mean()
        )
        return df['volume'] / (ma_vol + EPS)


# ==================== 3. LIQUIDITY & TURNOVER ====================

@FactorRegistry.register()
class TurnoverRate(TechnicalFactor):
    """
    Logarithmic Dollar Volume (Liquidity Proxy).
    Formula: ln(SMA(P × V, 21))
    """
    def __init__(self, period=21):
        super().__init__(
            name='turnover_rate',
            description='Log Dollar Volume (Liquidity)',
            lookback_period=period + 5
        )
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        dollar_vol = df['close'] * df['volume']
        return dollar_vol.groupby(df['ticker']).transform(
            lambda x: np.log(x.rolling(window=self.period).mean() + EPS)
        )


@FactorRegistry.register()
class Amihud63D(TechnicalFactor):
    """
    Amihud Illiquidity Ratio (63D).
    Formula: mean(|R| / P×V) × 1e6
    High = illiquid (large price impact per dollar volume).
    """
    def __init__(self, period=63):
        super().__init__(
            name='amihud_63d',
            description='Amihud Illiquidity 63D',
            lookback_period=period + 5
        )
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        dollar_vol   = (df['close'] * df['volume']) + EPS
        dollar_illiq = df['close'].pct_change().abs() / dollar_vol
        return dollar_illiq.groupby(df['ticker']).transform(
            lambda x: x.rolling(window=self.period).mean() * 1e6
        )


# ==================== 4. CORRELATION ====================

@FactorRegistry.register()
class PriceVolumeCorr21D(TechnicalFactor):
    """
    Price-Volume Correlation (21D).
    Positive ρ: volume expands on rallies (bullish).
    """
    def __init__(self, period=21):
        super().__init__(
            name='pv_corr_21d',
            description='Price-Volume Correlation 21D',
            lookback_period=period + 5
        )
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_corr(x):
            return x['close'].rolling(window=self.period).corr(x['volume'])

        return (
            df.groupby('ticker', group_keys=False)
              .apply(calc_corr, include_groups=False)
              .fillna(0)
        )


# ==================== 5. FORCE & EFFICIENCY ====================

@FactorRegistry.register()
class ForceIndex14D(TechnicalFactor):
    """
    Elder's Force Index (14D).
    Formula: EMA(ΔP × V, 14) / (mean dollar volume)
    """
    def __init__(self, period=14):
        super().__init__(
            name='force_index_14d',
            description='Elder Force Index 14D',
            lookback_period=period + 5
        )
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_force(x):
            raw_force  = x['close'].diff() * x['volume']
            ema_force  = raw_force.ewm(span=self.period, adjust=False).mean()
            norm_factor = (x['close'] * x['volume']).rolling(window=self.period).mean()
            return ema_force / (norm_factor + EPS)

        return (
            df.groupby('ticker', group_keys=False)
              .apply(calc_force, include_groups=False)
        )


# ==================== 6. CLASSIC FLOW OSCILLATORS ====================

@FactorRegistry.register()
class ChaikinMoneyFlow21D(TechnicalFactor):
    """
    Chaikin Money Flow (CMF).
    Formula: sum(MFV_21) / sum(V_21)
    """
    def __init__(self, period=21):
        super().__init__(
            name='cmf_21d',
            description='Chaikin Money Flow 21D',
            lookback_period=period + 5
        )
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_cmf(x):
            high = safe_col(x, "high")
            low  = safe_col(x, "low")
            if high.isna().all():
                return pd.Series(np.nan, index=x.index)

            range_len     = (high - low) + EPS
            mf_multiplier = ((x['close'] - low) - (high - x['close'])) / range_len
            mf_volume     = mf_multiplier * x['volume']

            rolling_mf_vol = mf_volume.rolling(window=self.period).sum()
            rolling_vol    = x['volume'].rolling(window=self.period).sum()
            return rolling_mf_vol / (rolling_vol + EPS)

        return (
            df.groupby('ticker', group_keys=False)
              .apply(calc_cmf, include_groups=False)
        )


@FactorRegistry.register()
class MoneyFlowIndex14(TechnicalFactor):
    """
    Money Flow Index (MFI) — volume-weighted RSI.
    """
    def __init__(self, period=14):
        super().__init__(
            name='mfi_14',
            description='Money Flow Index 14D',
            lookback_period=period + 5
        )
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_mfi(x):
            high = safe_col(x, "high")
            low  = safe_col(x, "low")
            if high.isna().all():
                return pd.Series(np.nan, index=x.index)

            tp      = (high + low + x['close']) / 3
            raw_mf  = tp * x['volume']
            tp_diff = tp.diff()

            pos_flow = raw_mf.where(tp_diff > 0, 0)
            neg_flow = raw_mf.where(tp_diff < 0, 0)

            pos_mf_sum = pos_flow.rolling(window=self.period).sum()
            neg_mf_sum = neg_flow.rolling(window=self.period).sum()

            mr  = pos_mf_sum / (neg_mf_sum + EPS)
            mfi = 100 - (100 / (1 + mr))
            return mfi

        return (
            df.groupby('ticker', group_keys=False)
              .apply(calc_mfi, include_groups=False)
        )


# ==================== 7. ACCUMULATION/DISTRIBUTION LINE ====================

@FactorRegistry.register()
class AccumulationDistribution(TechnicalFactor):
    """
    Accumulation/Distribution Line (A/D).
    Cumulative measure of volume flow based on the close's location within H-L range.

    FIX BUG-041: mfv_series was created as pd.Series(mfv) (0-indexed), losing
    the group's original index. This caused silent index misalignment when apply()
    reassembled results across groups with non-contiguous indices.
    Fixed: pd.Series(mfv, index=group.index) preserves the original row labels.
    """
    def __init__(self):
        super().__init__(
            name='ad_line',
            description='Accumulation/Distribution Line',
            lookback_period=2
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_ad(group):
            high  = safe_col(group, "high")
            low   = safe_col(group, "low")
            close = group['close']

            if high.isna().all():
                return pd.Series(np.nan, index=group.index)

            volume  = group['volume']
            range_hl = high - low

            mfm = np.where(
                range_hl != 0,
                ((close - low) - (high - close)) / range_hl,
                0,
            )

            # FIX BUG-041: preserve group.index so apply() reassembles correctly
            mfv_series   = pd.Series(mfm * volume.values, index=group.index)
            ad_line      = mfv_series.cumsum()
            ad_normalized = ad_line / (volume.sum() + EPS)
            return ad_normalized

        return (
            df.groupby('ticker', group_keys=False)
              .apply(calc_ad, include_groups=False)
        )