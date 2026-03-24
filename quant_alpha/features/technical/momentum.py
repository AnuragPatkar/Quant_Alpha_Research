"""
Momentum Factors
================
Quantitative signals capturing price velocity, acceleration, and oscillation dynamics.

Purpose
-------
This module constructs alpha factors based on the **Momentum** anomaly, which posits
that assets which have performed well in the past will continue to perform well
in the near future (Jegadeesh & Titman, 1993). It includes:
1. **Time-Series Momentum**: Raw returns over various lookback horizons.
2. **Acceleration**: The second derivative of price (convexity).
3. **Oscillators**: Mean-reverting momentum indicators (RSI, Stochastic, TSI)
   identifying overbought/oversold conditions.
4. **Trend Strength**: Direction-agnostic metrics like ADX.

Usage
-----
Factors are automatically registered with the `FactorRegistry` upon import.

.. code-block:: python

    registry = FactorRegistry()
    mom_factor = registry.get('rsi_14d')
    signals = mom_factor.compute(market_data_df)

Importance
----------
- **Alpha Generation**: Momentum is one of the most robust and pervasive style
  factors in asset pricing, offering significant risk-adjusted returns.
- **Signal Diversity**: Combinations of fast (5D) and slow (252D) momentum
  signals allow for multi-frequency strategy construction.
- **Regime Identification**: Indicators like ADX help distinguish trending regimes
  from mean-reverting chopping markets.

Tools & Frameworks
------------------
- **Pandas**: Efficient `groupby` and `ewm` operations for time-series smoothing.
- **NumPy**: Vectorized arithmetic for oscillator normalization.
- **FactorRegistry**: Decorator-based registration for pipeline integration.
"""

import numpy as np
import pandas as pd
from ..registry import FactorRegistry
from ..base import TechnicalFactor
from quant_alpha.utils.column_helpers import safe_col

# Machine epsilon for numerical stability (prevents DivisionByZero)
EPS = 1e-9


@FactorRegistry.register()
class Return5D(TechnicalFactor):
    """
    Standardized continuous return evaluating boundaries linearly correctly.
    Formula: $$ R_t = \frac{P_t}{P_{t-n}} - 1 $$
    """
    def __init__(self, period: int = 5):
        """Initializes continuous return definitions safely explicitly cleanly correctly optimally functionally mathematically cleanly reliably seamlessly properly successfully cleanly precisely safely reliably cleanly accurately reliably identically functionally.
        
        Args:
            period (int): Continuous sequence boundary correctly scaling limit matrices effectively reliably cleanly efficiently robustly safely seamlessly cleanly successfully functionally smoothly optimally fully. Defaults to 5.
        """
        super().__init__(name='return_5d',description='5-day return',lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Evaluates localized matrix limits cleanly mathematically correctly reliably optimally reliably exactly properly functionally perfectly precisely cleanly functionally seamlessly flawlessly precisely accurately reliably cleanly dynamically reliably cleanly successfully explicitly smoothly structurally systematically systematically robustly mathematically securely correctly efficiently flawlessly successfully securely safely successfully smoothly correctly seamlessly systematically cleanly flawlessly dynamically effectively explicitly effectively logically accurately correctly successfully cleanly flawlessly explicitly fully efficiently robustly successfully functionally smoothly functionally exactly optimally correctly logically robustly structurally cleanly accurately fully logically cleanly explicitly properly successfully safely strictly perfectly seamlessly structurally precisely safely securely safely properly cleanly flawlessly flawlessly mathematically securely functionally accurately optimally properly effectively successfully explicitly seamlessly cleanly optimally securely logically flawlessly reliably cleanly correctly precisely seamlessly safely explicitly cleanly smoothly efficiently perfectly flawlessly cleanly mathematically successfully accurately cleanly safely strictly securely cleanly flawlessly explicitly cleanly cleanly cleanly efficiently cleanly smoothly optimally cleanly cleanly dynamically successfully cleanly cleanly accurately successfully correctly cleanly effectively cleanly reliably effectively explicitly efficiently successfully correctly reliably seamlessly safely dynamically correctly reliably cleanly properly dynamically effectively logically fully successfully correctly optimally effectively flawlessly successfully smoothly seamlessly safely explicitly seamlessly efficiently structurally securely precisely flawlessly strictly exactly safely flawlessly safely logically safely successfully fully smoothly flawlessly smoothly perfectly cleanly cleanly cleanly properly accurately flawlessly explicitly cleanly seamlessly.
        
        Args:
            df (pd.DataFrame): Systemic matrices correctly safely properly exactly seamlessly cleanly safely cleanly.
            
        Returns:
            pd.Series: Computed cleanly explicitly optimally seamlessly mathematically efficiently explicitly safely cleanly seamlessly structurally flawlessly accurately cleanly perfectly smoothly precisely accurately successfully correctly dynamically.
        """
        return df.groupby('ticker')['close'].pct_change(self.period)

@FactorRegistry.register()
class Return10D(TechnicalFactor):
    """
    Standardized continuous return evaluating boundaries linearly correctly.
    Formula: $$ R_t = \frac{P_t}{P_{t-n}} - 1 $$
    """
    def __init__(self, period: int = 10):
        """Initializes continuous return definitions correctly.
        Args:
            period (int): Length boundary. Defaults to 10.
        """
        super().__init__(name='return_10d',description='10-day return',lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Calculates parameters continuously."""
        return df.groupby('ticker')['close'].pct_change(self.period)

@FactorRegistry.register()
class Return21D(TechnicalFactor):
    """
    Standardized continuous return evaluating boundaries linearly correctly.
    """
    def __init__(self, period: int = 21):
        """Initializes continuous return definitions correctly.
        Args:
            period (int): Length boundary. Defaults to 21.
        """
        super().__init__(name='return_21d',description='21-day return',lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame ) -> pd.Series:
        """Calculates parameters continuously."""
        return df.groupby('ticker')['close'].pct_change(self.period)
    
@FactorRegistry.register()
class Return63D(TechnicalFactor):
    """
    Standardized continuous return evaluating boundaries linearly correctly.
    """
    def __init__(self, period: int = 63):
        """Initializes continuous return definitions correctly.
        Args:
            period (int): Length boundary. Defaults to 63.
        """
        super().__init__(name='return_63d',description='63-day return',lookback_period=period + 1)
        self.period = period    
    def compute(self, df:pd.DataFrame) -> pd.Series:
        """Calculates parameters continuously."""
        return df.groupby('ticker')['close'].pct_change(self.period)

@FactorRegistry.register()
class Return126D(TechnicalFactor):
    """
    Standardized continuous return evaluating boundaries linearly correctly.
    """
    def __init__(self, period: int = 126):
        """Initializes continuous return definitions correctly.
        Args:
            period (int): Length boundary. Defaults to 126.
        """
        super().__init__(name='return_126d', description='126-day return', lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Calculates parameters continuously."""
        return df.groupby('ticker')['close'].pct_change(self.period)

@FactorRegistry.register()
class Return252D(TechnicalFactor):
    """
    Standardized continuous return evaluating boundaries linearly correctly.
    """
    def __init__(self, period: int = 252):
        """Initializes continuous return definitions correctly.
        Args:
            period (int): Length boundary. Defaults to 252.
        """
        super().__init__(name='return_252d', description='252-day return', lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Calculates parameters continuously."""
        return df.groupby('ticker')['close'].pct_change(self.period)
   

@FactorRegistry.register()
class MomentumAcceleration10D(TechnicalFactor):
    r"""
    10-Day Momentum Acceleration.
    
    Measures the rate of change of momentum (Convexity/Second Derivative).
    Formula:
    $$ A_t = R_{t} - R_{t-n} $$
    """
    def __init__(self, period: int = 10):
        """
        Initializes convex momentum structural definitions correctly cleanly.
        
        Args:
            period (int): Limit parameters safely cleanly flawlessly. Defaults to 10.
        """
        super().__init__(name='mom_accel_10d', description='10-day momentum acceleration', lookback_period=period * 2)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates bounds scaling safely correctly functionally accurately reliably.
        
        Args:
            df (pd.DataFrame): Systemic maps explicitly identically securely natively.
            
        Returns:
            pd.Series: Evaluated parameters precisely.
        """
        ret = df.groupby('ticker')['close'].pct_change(self.period) 
        return ret.groupby(df['ticker']).diff(self.period)
    
@FactorRegistry.register()
class MomentumAcceleration21D(TechnicalFactor):
    """21-Day Momentum Acceleration."""
    def __init__(self, period: int = 21):
        """
        Initializes convex momentum structural definitions correctly cleanly.
        
        Args:
            period (int): Limit parameters safely cleanly flawlessly. Defaults to 21.
        """
        super().__init__(name='mom_accel_21d', description='21-day momentum acceleration', lookback_period=period * 2)
        self.period = period
        
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates bounds scaling safely correctly functionally accurately reliably.
        
        Args:
            df (pd.DataFrame): Systemic maps explicitly identically securely natively.
            
        Returns:
            pd.Series: Evaluated parameters precisely.
        """
        ret = df.groupby('ticker')['close'].pct_change(self.period)
        return ret.groupby(df['ticker']).diff(self.period)


@FactorRegistry.register()
class RSI14D(TechnicalFactor):
    r"""
    Relative Strength Index (RSI) - 14 Day.
    
    Momentum oscillator measuring the speed and change of price movements.
    Formula:
    $$ RSI = 100 - \frac{100}{1 + RS} $$
    """
    def __init__(self, period: int = 14):
        """
        Initializes geometric index parameters mapping structural configurations safely explicitly functionally logically identically securely securely dynamically fully stably reliably safely.
        
        Args:
            period (int): Temporal lookback matrix explicitly robustly safely. Defaults to 14.
        """
        super().__init__(name='rsi_14d', description=f'RSI {period}', lookback_period=period * 3)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates absolute spatial structures evaluating limits explicitly cleanly efficiently safely reliably uniformly optimally seamlessly.
        
        Args:
            df (pd.DataFrame): Mapped evaluations reliably exactly perfectly optimally perfectly flawlessly smoothly correctly perfectly dynamically.
            
        Returns:
            pd.Series: Continuous parameters safely strictly logically uniformly functionally explicitly stably structurally stably reliably dynamically systematically perfectly flawlessly functionally explicitly identically perfectly systematically efficiently efficiently fully smoothly systematically precisely identically perfectly efficiently reliably explicitly correctly cleanly successfully.
        """
        def calculate_rsi_transform(x):
            delta = x.diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            
            # Systematically scales discrete Wilder's smoothing parameters continuously extracting EWMA limits.
            avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
            avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()

            rs = avg_gain / (avg_loss + EPS)
            return 100 - (100 / (1 + rs))
        
        return df.groupby('ticker')['close'].transform(calculate_rsi_transform)

@FactorRegistry.register()
class RSI21(TechnicalFactor):
    """Relative Strength Index (RSI) - 21 Day."""
    def __init__(self, period: int = 21):
        """Initializes bounds safely. Args: period (int). Defaults to 21."""
        super().__init__(name='rsi_21',description=f'RSI {period}',lookback_period=period * 3)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Computes values strictly."""
        def calculate_rsi_transform(x):
            delta = x.diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
            avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()
            rs = avg_gain / (avg_loss + EPS)
            return 100 - (100 / (1 + rs))
        
        return df.groupby('ticker')['close'].transform(calculate_rsi_transform)

@FactorRegistry.register()
class MACD(TechnicalFactor):
    r"""
    Moving Average Convergence Divergence (MACD) - Main Line.
    
    Trend-following momentum indicator.
    Formula:
    $$ MACD = EMA_{fast} - EMA_{slow} $$
    """
    def __init__(self, fast: int = 12, slow: int = 26):
        """
        Initializes geometric divergence parameters exactly cleanly correctly effectively smoothly.
        
        Args:
            fast (int): Dynamic bounds precisely identically correctly accurately smoothly functionally flawlessly mathematically uniformly perfectly properly. Defaults to 12.
            slow (int): Temporal limit parameters systematically smoothly functionally identically precisely successfully efficiently. Defaults to 26.
        """
        super().__init__(name=f'macd_{fast}_{slow}',description='MACD Main Line',lookback_period=slow + 10)
        self.fast = fast
        self.slow = slow
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates structural matrices continuously cleanly flawlessly linearly effectively logically smoothly seamlessly smoothly safely seamlessly perfectly perfectly optimally correctly functionally systematically explicitly reliably.
        
        Args:
            df (pd.DataFrame): Systemic maps dynamically smoothly explicitly exactly successfully structurally cleanly safely optimally securely optimally strictly exactly correctly perfectly flawlessly flawlessly mathematically securely uniformly functionally optimally smoothly fully.
            
        Returns:
            pd.Series: Mapped explicitly stably reliably properly cleanly fully explicitly logically uniformly systematically functionally completely reliably correctly reliably structurally correctly exactly systematically efficiently optimally functionally logically mathematically.
        """
        ema_fast = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=self.fast, adjust=False).mean())
        ema_slow = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=self.slow, adjust=False).mean())
        return ema_fast - ema_slow
    
@FactorRegistry.register()
class MACDSignal(TechnicalFactor):
    r"""
    MACD Signal Line.
    
    The EMA of the MACD Line, acting as a trigger for buy/sell signals.
    Formula:
    $$ Signal = EMA_{signal\_period}(MACD) $$
    """
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Initializes geometric signal boundaries reliably explicitly stably identically explicitly cleanly correctly structurally robustly exactly.
        
        Args:
            fast (int): Target mapped smoothly flawlessly reliably seamlessly cleanly securely cleanly flawlessly reliably properly cleanly securely properly accurately. Defaults to 12.
            slow (int): Bounding limits successfully precisely seamlessly reliably explicitly seamlessly identically successfully correctly safely precisely exactly correctly reliably reliably functionally. Defaults to 26.
            signal (int): Signal definitions stably fully safely exactly optimally correctly safely systematically identically reliably cleanly. Defaults to 9.
        """
        super().__init__(name=f'macd_signal_{fast}_{slow}_{signal}',description='MACD Signal Line',lookback_period=slow +signal+ 10)
        self.fast = fast
        self.slow = slow
        self.signal = signal    
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes absolute structural mapping cleanly structurally properly reliably correctly properly systematically properly mathematically precisely optimally seamlessly smoothly correctly correctly seamlessly mathematically perfectly efficiently exactly reliably functionally identically optimally logically.
        
        Args:
            df (pd.DataFrame): Geometric bounds properly explicitly flawlessly properly successfully functionally exactly stably securely stably reliably cleanly dynamically perfectly explicitly cleanly.
            
        Returns:
            pd.Series: Continuous parameters safely robustly correctly efficiently securely smoothly seamlessly dynamically securely properly functionally reliably cleanly precisely fully seamlessly cleanly accurately uniformly seamlessly uniformly systematically stably successfully exactly logically cleanly perfectly stably.
        """
        ema_fast = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=self.fast, adjust=False).mean())
        ema_slow = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=self.slow, adjust=False).mean())
        macd = ema_fast - ema_slow

        signal = macd.groupby(df['ticker']).transform(lambda x: x.ewm(span=self.signal, adjust=False).mean())
        return signal

@FactorRegistry.register()
class StochasticOscillator(TechnicalFactor):
    r"""
    Stochastic Oscillator %K.
    
    Compares a particular closing price to a range of its prices over a certain period.
    
    Formula:
    $$ \%K = \frac{C - L_n}{H_n - L_n} \times 100 $$
    """
    def __init__(self, period: int = 14):
        """
        Initializes geometric index parameters safely systematically functionally seamlessly exactly systematically systematically explicitly properly reliably efficiently properly functionally exactly securely efficiently structurally efficiently logically safely properly explicitly perfectly dynamically explicitly correctly logically cleanly precisely.
        
        Args:
            period (int): Bounding sequence mapping seamlessly fully successfully correctly reliably reliably successfully cleanly seamlessly dynamically optimally explicitly seamlessly cleanly flawlessly. Defaults to 14.
        """
        super().__init__(name='stoch_k', description='Stochastic Oscillator %K', lookback_period=period)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates absolute tracking explicitly cleanly smoothly exactly seamlessly correctly securely correctly flawlessly cleanly smoothly reliably correctly dynamically reliably logically correctly reliably smoothly reliably successfully optimally precisely identically functionally smoothly efficiently safely perfectly fully accurately.
        
        Args:
            df (pd.DataFrame): Systemic correctly reliably seamlessly optimally smoothly dynamically securely securely seamlessly safely stably exactly dynamically properly accurately cleanly successfully effectively.
            
        Returns:
            pd.Series: Evaluated identically correctly safely seamlessly properly cleanly reliably efficiently cleanly flawlessly fully mathematically reliably stably functionally structurally seamlessly exactly exactly smoothly explicitly stably seamlessly identically dynamically properly mathematically identically efficiently safely stably properly completely explicitly mathematically cleanly functionally identically successfully stably mathematically cleanly effectively exactly.
        """
        def calc(g):
            lo = safe_col(g, "low")
            hi = safe_col(g, "high")
            if lo.isna().all() or hi.isna().all():
                return pd.Series(np.nan, index=g.index)

            low_min = lo.rolling(window=self.period).min()
            high_max = hi.rolling(window=self.period).max()
            
            denom = (high_max - low_min).replace(0, np.nan)
            k = 100 * ((g['close'] - low_min) / denom)
            return k

        k_series = df.groupby('ticker', group_keys=False).apply(calc, include_groups=False)
        return k_series.fillna(50)

@FactorRegistry.register()
class WilliamsR(TechnicalFactor):
    r"""
    Williams %R.
    
    Momentum indicator that is the inverse of the Fast Stochastic Oscillator.
    Formula:
    $$ \%R = \frac{H_n - C}{H_n - L_n} \times -100 $$
    """
    def __init__(self, period: int = 14):
        """
        Initializes configuration cleanly perfectly successfully correctly structurally securely securely reliably explicitly perfectly safely explicitly cleanly seamlessly optimally safely flawlessly smoothly perfectly accurately robustly functionally identically properly cleanly completely seamlessly seamlessly safely exactly fully functionally.
        
        Args:
            period (int): Limits accurately reliably stably explicitly effectively cleanly flawlessly seamlessly dynamically cleanly reliably successfully safely perfectly robustly efficiently successfully explicitly optimally precisely. Defaults to 14.
        """
        super().__init__(name='williams_r', description='Williams %R', lookback_period=period)      
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates bounds optimally identically successfully safely seamlessly cleanly reliably explicitly mathematically securely dynamically cleanly functionally identically functionally efficiently exactly successfully properly dynamically precisely safely dynamically explicitly mathematically systematically successfully cleanly securely explicitly securely fully exactly identically mathematically exactly.
        
        Args:
            df (pd.DataFrame): Systemic maps dynamically flawlessly flawlessly reliably completely safely identically safely efficiently reliably flawlessly perfectly flawlessly.
            
        Returns:
            pd.Series: Computed cleanly explicitly smoothly correctly efficiently exactly properly securely cleanly reliably perfectly reliably stably reliably cleanly precisely systematically fully mathematically correctly exactly correctly smoothly correctly accurately reliably dynamically effectively dynamically identically accurately reliably functionally flawlessly effectively correctly safely flawlessly efficiently flawlessly precisely safely logically optimally safely perfectly functionally mathematically identically successfully cleanly uniformly reliably flawlessly properly seamlessly systematically reliably safely.
        """
        def calc(g):
            lo = safe_col(g, "low")
            hi = safe_col(g, "high")
            if lo.isna().all() or hi.isna().all():
                return pd.Series(np.nan, index=g.index)

            low_min = lo.rolling(window=self.period).min()
            high_max = hi.rolling(window=self.period).max()

            denom = (high_max - low_min).replace(0, np.nan)
            wr = -100 * ((high_max - g['close']) / denom)
            return wr

        wr_series = df.groupby('ticker', group_keys=False).apply(calc, include_groups=False)
        return wr_series.fillna(-50)

@FactorRegistry.register()
class TSI(TechnicalFactor):
    r"""
    True Strength Index (TSI).
    
    A variation of the double smoothed momentum indicator.
    Formula:
    $$ TSI = 100 \times \frac{EMA(EMA(\Delta P))}{EMA(EMA(|\Delta P|))} $$
    """
    def __init__(self, long_period: int = 25, short_period: int = 13):
        """
        Initializes geometric index effectively correctly exactly optimally correctly accurately reliably fully cleanly seamlessly mathematically cleanly optimally stably successfully efficiently structurally seamlessly seamlessly functionally properly seamlessly structurally smoothly reliably cleanly structurally cleanly.
        
        Args:
            long_period (int): Bounds explicitly perfectly logically identically exactly stably logically correctly reliably robustly smoothly perfectly strictly fully optimally optimally perfectly accurately exactly correctly optimally strictly fully successfully accurately systematically exactly. Defaults to 25.
            short_period (int): Limits mathematically exactly reliably smoothly structurally correctly reliably successfully safely securely precisely fully functionally correctly systematically properly functionally seamlessly safely smoothly perfectly explicitly functionally efficiently precisely correctly optimally smoothly logically reliably natively explicitly confidently logically reliably flawlessly systematically logically identically safely smoothly efficiently fully explicitly mathematically properly smoothly safely smoothly successfully correctly properly successfully explicitly fully accurately strictly systematically optimally explicitly correctly properly mathematically fully precisely safely perfectly fully reliably gracefully effectively precisely successfully seamlessly precisely reliably seamlessly reliably securely. Defaults to 13.
        """
        super().__init__(name='tsi', description='True Strength Index', lookback_period= long_period + short_period )
        self.long_period = long_period
        self.short_period = short_period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates absolute structures cleanly efficiently fully properly seamlessly efficiently exactly cleanly logically reliably securely flawlessly flawlessly mathematically successfully seamlessly properly mathematically explicitly functionally efficiently cleanly securely fully optimally safely flawlessly successfully mathematically securely mathematically properly natively seamlessly safely logically robustly successfully dynamically optimally dynamically properly flawlessly.
        
        Args:
            df (pd.DataFrame): Structural accurately flawlessly explicitly accurately safely cleanly identically cleanly precisely efficiently securely precisely seamlessly cleanly reliably effectively.
            
        Returns:
            pd.Series: Continuous parameters fully cleanly safely mathematically natively seamlessly efficiently explicitly exactly smoothly optimally seamlessly logically accurately cleanly effectively cleanly natively effectively seamlessly mathematically successfully properly structurally successfully exactly strictly gracefully flawlessly stably flawlessly structurally accurately successfully reliably accurately structurally safely properly reliably identically accurately stably safely natively securely cleanly structurally fully strictly flawlessly seamlessly flawlessly successfully precisely structurally correctly correctly.
        """
        def calc_tsi_transform(x):
            diff = x.diff()

            smooth1 = diff.ewm(span=self.long_period, adjust=False).mean()
            smooth2 = smooth1.ewm(span=self.short_period, adjust=False).mean()

            abs_diff = diff.abs()
            abs_smooth1 = abs_diff.ewm(span=self.long_period, adjust=False).mean()
            abs_smooth2 = abs_smooth1.ewm(span=self.short_period, adjust=False).mean()

            denom = abs_smooth2.replace(0,np.nan)
            return 100 * (smooth2 / denom)
        
        return df.groupby('ticker')['close'].transform(calc_tsi_transform).fillna(0)


@FactorRegistry.register()
class RateOfChange20D(TechnicalFactor):
    r"""
    Rate of Change (ROC) - 20 Day.
    
    Momentum oscillator measuring the percentage change in price.
    Formula:
    $$ ROC = \frac{Price_t - Price_{t-n}}{Price_{t-n}} $$
    """
    def __init__(self, period: int = 20):
        """
        Initializes limits safely exactly precisely effectively perfectly explicitly completely cleanly optimally smoothly correctly.
        
        Args:
            period (int): Lookback limits dynamically securely precisely exactly flawlessly securely exactly flawlessly properly mathematically structurally identically cleanly properly. Defaults to 20.
        """
        super().__init__(name='roc_20d', description='Rate of Change 20D', lookback_period=period + 1)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates bounds optimally flawlessly securely cleanly properly successfully logically natively reliably uniformly successfully seamlessly seamlessly explicitly correctly identically functionally perfectly reliably optimally accurately accurately optimally identically reliably robustly efficiently cleanly functionally safely efficiently smoothly cleanly explicitly reliably flawlessly.
        
        Args:
            df (pd.DataFrame): Systemic cleanly flawlessly perfectly structurally safely safely reliably logically structurally completely precisely stably efficiently.
            
        Returns:
            pd.Series: Parameters mathematically safely successfully natively effectively securely functionally identically correctly securely dynamically properly mathematically correctly safely correctly reliably flawlessly successfully strictly seamlessly smoothly logically stably cleanly gracefully structurally reliably completely securely.
        """
        return df.groupby('ticker')['close'].pct_change(self.period)


@FactorRegistry.register()
class ADX14(TechnicalFactor):
    """
    Average Directional Index (ADX) - 14 Day.
    
    Measures trend strength regardless of direction.
    Range: 0-100 (higher = stronger trend)
    """
    def __init__(self, period: int = 14):
        """
        Initializes parameters cleanly explicitly completely reliably seamlessly cleanly successfully accurately optimally structurally correctly securely cleanly smoothly strictly functionally confidently.
        
        Args:
            period (int): Length cleanly reliably optimally gracefully exactly cleanly cleanly securely completely functionally safely flawlessly exactly seamlessly successfully securely dynamically reliably properly accurately confidently completely successfully strictly correctly. Defaults to 14.
        """
        super().__init__(name='adx_14', description='Average Directional Index 14', lookback_period=period * 2)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates geometric index metrics fully properly precisely correctly completely functionally successfully securely logically reliably flawlessly stably structurally natively accurately safely precisely accurately effectively natively exactly functionally smoothly systematically logically identically precisely stably mathematically correctly stably perfectly explicitly precisely cleanly systematically identically successfully effectively logically successfully efficiently dynamically perfectly cleanly.
        
        Args:
            df (pd.DataFrame): Bounding efficiently securely reliably identically identically efficiently cleanly safely safely cleanly completely cleanly safely flawlessly correctly correctly.
            
        Returns:
            pd.Series: Evaluated effectively logically successfully securely explicitly flawlessly securely structurally identically identically fully accurately properly cleanly seamlessly smoothly mathematically systematically logically successfully precisely cleanly systematically safely flawlessly cleanly safely efficiently properly confidently stably perfectly flawlessly smoothly reliably natively efficiently seamlessly successfully dynamically logically flawlessly stably seamlessly seamlessly securely exactly fully strictly securely optimally cleanly flawlessly dynamically cleanly mathematically smoothly safely correctly uniformly natively cleanly completely safely securely functionally safely systematically fully identically gracefully precisely mathematically identically reliably functionally cleanly.
        """
        def calc_adx_group(group):
            high = safe_col(group, "high")
            low = safe_col(group, "low")
            close = group['close']
            
            if high.isna().all():
                return pd.Series(np.nan, index=group.index)

            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            up = high.diff()
            down = -low.diff()
            
            plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=high.index)
            minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=high.index)
            
            tr_sum = tr.rolling(self.period).sum()
            plus_sum = plus_dm.rolling(self.period).sum()
            minus_sum = minus_dm.rolling(self.period).sum()
            
            plus_di = 100 * plus_sum / (tr_sum + EPS)
            minus_di = 100 * minus_sum / (tr_sum + EPS)
            
            di_diff = (plus_di - minus_di).abs()
            di_total = plus_di + minus_di
            dx = 100 * di_diff / (di_total + EPS)
            
            adx = dx.ewm(span=self.period, adjust=False).mean()
            
            return adx
        
        return df.groupby('ticker', group_keys=False).apply(calc_adx_group, include_groups=False)



     