"""
System Health & Regime Composite Factors (5 Factors)
Focus: Market regime classification, portfolio health scoring, system-level indicators.

Factors (Ranked by Importance):
1. MarketRegimeScore -> Bull/Bear/Sideways classification
2. VolatilityRegime -> Normal/Elevated/Crisis volatility levels
3. CapitalFlowSignal -> Oil + USD combined flow indication
4. EconomicMomentumScore -> Combined macro strength indicator
5. PortfolioHealthIndex -> Overall system health (0-100 scale)
"""

import pandas as pd
import numpy as np
from ..base import CompositeFactor
from ..registry import FactorRegistry
from config.logging_config import logger


@FactorRegistry.register()
class MarketRegimeScore(CompositeFactor):
    """
    Market Regime Score: Classify market as Bull/Bear/Sideways.
    Formula: +1 if (Ret_positive AND Trend_up) else -1 if (Ret_negative AND Trend_down) else 0
    
    Why: Regime determines which factors work (momentum in bull, mean-reversion in bear).
    Useful for dynamic factor allocation and hedge ratios.
    """
    def __init__(self):
        super().__init__(name='comp_regime', description='Market Regime Classification')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'sp500_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing SP500 data")
            return pd.Series(np.nan, index=df.index)
        
        # Recent performance (21-day)
        returns = df['sp500_close'].pct_change(21)
        
        # Trend (200-day moving average)
        trend = df['sp500_close'] > df['sp500_close'].rolling(200).mean()
        
        # Regime classification
        regime = pd.Series(0, index=df.index)
        regime[(returns > 0) & trend] = 1  # Bull regime
        regime[(returns < 0) & ~trend] = -1  # Bear regime
        
        return regime.fillna(0)


@FactorRegistry.register()
class VolatilityRegime(CompositeFactor):
    """
    Volatility Regime: Classify volatility as Normal/Elevated/Crisis.
    Formula: 0 if VIX < 15, 1 if 15 < VIX < 25, 2 if VIX > 25
    
    Why: Volatility regime drives optimal leverage, position sizing, and strategies.
    Crisis mode (VIX > 25) requires defensive positioning.
    Normal mode (VIX < 15) supports more aggressive strategies.
    """
    def __init__(self):
        super().__init__(name='comp_vol_regime', description='Volatility Regime Classification')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing VIX data")
            return pd.Series(np.nan, index=df.index)
        
        vix = df['vix_close'].rolling(5).mean()
        
        # Regime classification
        regime = pd.Series(0, index=df.index)
        regime[vix >= 15] = 1  # Elevated
        regime[vix >= 25] = 2  # Crisis
        
        return regime.fillna(0)


@FactorRegistry.register()
class CapitalFlowSignal(CompositeFactor):
    """
    Capital Flow Signal: Oil + USD trend indicating capital movements.
    Formula: (Oil_momentum + USD_momentum) / 2, normalized 0-100
    
    Why: Rising oil + rising USD = Risk-on (capital chasing growth).
    Falling oil + falling USD = Risk-off (capital seeking safety).
    Captures institutional flow patterns.
    """
    def __init__(self):
        super().__init__(name='comp_capital_flow', description='Capital Flow Signal Indicator')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        required = ['oil_close', 'usd_close']
        if not all(col in df.columns for col in required):
            logger.warning(f"❌ {self.name}: Missing oil or USD data")
            return pd.Series(np.nan, index=df.index)
        
        # Oil and USD momentum (21-day)
        oil_mom = df['oil_close'].pct_change(21)
        usd_mom = df['usd_close'].pct_change(21)
        
        # Normalize and combine
        oil_norm = (oil_mom.rolling(63).mean() + 0.05) / 0.10  # Scale to ~0-1
        usd_norm = (usd_mom.rolling(63).mean() + 0.05) / 0.10
        
        # Combined flow signal (0-100 scale)
        flow = ((oil_norm + usd_norm) / 2).clip(lower=0, upper=1) * 100
        
        return flow.fillna(50)


@FactorRegistry.register()
class EconomicMomentumScore(CompositeFactor):
    """
    Economic Momentum Score: Combined macro strength from yields + oil + USD.
    Formula: (Yield_momentum + Oil_momentum - USD_momentum) / 3, normalized 0-100
    
    Why: Rising yields + rising oil = Strong growth + inflation (macro strength).
    Rising yields + falling oil = Real growth leading inflation (best combo).
    Useful for tactical macro rotation and asset allocation.
    """
    def __init__(self):
        super().__init__(name='comp_econ_momentum', description='Economic Momentum Score (0-100)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        required = ['us_10y_close', 'oil_close', 'usd_close']
        if not all(col in df.columns for col in required):
            logger.warning(f"❌ {self.name}: Missing macro data")
            return pd.Series(np.nan, index=df.index)
        
        # Momentum components (21-day)
        yield_mom = df['us_10y_close'].pct_change(21)
        oil_mom = df['oil_close'].pct_change(21)
        usd_mom = df['usd_close'].pct_change(21)
        
        # Composite: Growth (yield) + Inflation (oil) - Currency (USD)
        # Positive USD = Headwind (strength kills exports)
        composite = (yield_mom + oil_mom - usd_mom) / 3
        
        # Normalize to 0-100 scale
        composite_norm = ((composite + 0.05) / 0.10).clip(lower=0, upper=1) * 100
        
        return composite_norm.fillna(50)


@FactorRegistry.register()
class PortfolioHealthIndex(CompositeFactor):
    """
    Portfolio Health Index: Overall system health synthesis (0-100 scale).
    Formula: (Market_regime_positive * 25 + Vol_regime_normal * 25 + 
              Flow_signal * 0.25 + Econ_momentum * 0.25) / 100 * 100
    
    Why: Single metric for portfolio managers to assess system health.
    100 = Perfect conditions (bull market, low vol, positive flow, strong macro).
    0 = Crisis conditions (bear market, high vol, negative flow, weak macro).
    Useful for risk limits and hedge ratio decisions.
    """
    def __init__(self):
        super().__init__(name='comp_health_index', description='Portfolio Health Index (0-100)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'sp500_close' not in df.columns or 'vix_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing critical data")
            return pd.Series(np.nan, index=df.index)
        
        # Market regime component (0-25 scale)
        returns = df['sp500_close'].pct_change(21)
        trend = df['sp500_close'] > df['sp500_close'].rolling(200).mean()
        regime_score = ((returns > 0) & trend).astype(float) * 25
        
        # Volatility regime component (0-25 scale, inverse)
        vix = df['vix_close'].rolling(5).mean()
        vol_score = ((40 - vix) / 30).clip(lower=0, upper=1) * 25
        
        # Flow signal component (0-25 scale)
        if 'oil_close' in df.columns and 'usd_close' in df.columns:
            oil_mom = df['oil_close'].pct_change(21)
            usd_mom = df['usd_close'].pct_change(21)
            flow_score = ((oil_mom + usd_mom) / 2).rolling(21).mean() * 250  # Scale to 0-25
            flow_score = flow_score.clip(lower=0, upper=25)
        else:
            flow_score = pd.Series(12.5, index=df.index)  # Default neutral
        
        # Economic momentum component (0-25 scale)
        if 'us_10y_close' in df.columns:
            yield_mom = df['us_10y_close'].pct_change(21)
            econ_score = ((yield_mom + 0.05) / 0.10).clip(lower=0, upper=1) * 25
        else:
            econ_score = pd.Series(12.5, index=df.index)  # Default neutral
        
        # Combine all components
        health_index = regime_score + vol_score + flow_score + econ_score
        
        return health_index.fillna(50)
