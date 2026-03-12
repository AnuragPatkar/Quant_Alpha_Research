"""
Backtesting Utilities & Shared Primitives
=========================================
Foundational logic for data alignment, financial calculus, and schema validation.

Purpose
-------
This module serves as the functional bedrock for the backtesting engine. It centralizes
critical, repetitive operations—such as return calculation, date alignment, and data
sanitization—ensuring mathematical consistency and type safety across the platform.
Optimized for vectorization, it handles the "plumbing" so higher-level modules can
focus on strategy logic.

Usage
-----
.. code-block:: python

    from quant_alpha.backtest.utils import calculate_returns, validate_backtest_data

    # Compute log returns for stationarity
    log_rets = calculate_returns(prices['close'], method='log')

    # Ensure signal data integrity before execution
    is_valid, report = validate_backtest_data(signals, prices)

Importance
----------
- **Data Integrity**: Enforces strict schema validation ($O(N)$) to prevent silent failures in matrix operations.
- **Financial Precision**: Provides mathematically rigorous implementations of core metrics (e.g., $SR$, $MDD$) and return transformations ($R_t = \ln(P_t / P_{t-1})$).
- **Performance**: Utilizes vectorized Pandas/NumPy operations to handle large panel datasets efficiently, minimizing copy overhead during date alignment.

Tools & Frameworks
------------------
- **Pandas**: Core data structure for time-series alignment and rolling calculations.
- **NumPy**: Vectorized arithmetic for high-performance financial math.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ==================== RETURN CALCULATIONS ====================

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculates periodic returns for time-series data.

    Args:
        prices: Price series.
        method: 'simple' ($R_t = \\frac{P_t}{P_{t-1}} - 1$) or 'log' ($R_t = \\ln(\\frac{P_t}{P_{t-1}})$).
    """
    if method == 'log':
        return np.log(prices / prices.shift(1))
    return prices.pct_change()

def cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Computes the cumulative equity growth path (Wealth Index).
    
    Formula: $V_t = V_0 \\times \\prod_{i=1}^t (1 + r_i)$
    """
    return (1 + returns.fillna(0)).cumprod()

# ==================== DATE OPERATIONS (OPTIMIZED) ====================

def align_dates(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synchronizes two DataFrames to their intersection of dates.

    Uses strict set intersection ($O(M+N)$) to ensure point-in-time alignment,
    critical for correlation analysis and signal processing.
    """
    if df1.empty or df2.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Optimization: Type conversion performed once to prevent repetitive casting overhead.
    d1_temp = pd.to_datetime(df1['date'])
    d2_temp = pd.to_datetime(df2['date'])
    
    # Intersection of unique dates prevents Cartesian explosion in join operations.
    common_dates = sorted(list(set(d1_temp).intersection(set(d2_temp))))
    
    if not common_dates:
        logger.warning("No common dates found between datasets.")
        return pd.DataFrame(), pd.DataFrame()
        
    if len(common_dates) < len(d1_temp):
        logger.info(f"📉 Date Alignment: Dropped {len(d1_temp) - len(common_dates)} rows from DF1 to match dates.")

    # Vectorized boolean masking using pre-converted series for efficiency.
    df1_aligned = df1[d1_temp.isin(common_dates)].copy()
    df2_aligned = df2[d2_temp.isin(common_dates)].copy()
    
    return df1_aligned, df2_aligned

# ==================== DATA VALIDATION (HARDENED) ====================

def validate_backtest_data(predictions: pd.DataFrame, prices: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Enforces schema integrity and numeric type safety for backtest inputs.

    Validates:
    1. **Schema**: Presence of required columns (`date`, `ticker`, value cols).
    2. **Types**: Numeric constraints to prevent object-dtype aggregation errors.
    3. **Uniqueness**: Composite key (`date`, `ticker`) integrity for pivoting.
    """
    errors = []
    
    # 1. Schema Validation (Column Existence)
    pred_req = {'date', 'ticker', 'prediction'}
    price_req = {'date', 'ticker', 'close'}
    
    if not pred_req.issubset(predictions.columns):
        errors.append(f"Predictions missing: {pred_req - set(predictions.columns)}")
    if not price_req.issubset(prices.columns):
        errors.append(f"Prices missing: {price_req - set(prices.columns)}")
    
    if errors: return False, errors

    # 2. Type Safety (Numeric Integrity)
    if not pd.api.types.is_numeric_dtype(predictions['prediction']):
        errors.append("Prediction column must be numeric.")
    
    if not pd.api.types.is_numeric_dtype(prices['close']):
        errors.append("Price close column must be numeric.")

    # 3. Key Integrity (Duplicate Detection)
    if predictions.duplicated(subset=['date', 'ticker']).any():
        errors.append("Duplicate (date, ticker) pairs in predictions found.")

    return len(errors) == 0, errors

# ==================== PERFORMANCE HELPERS ====================

def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02, periods: int = 252) -> float:
    """
    Computes the Annualized Sharpe Ratio.

    Formula: $SR = \\frac{E[R_p - R_f]}{\\sigma_p} \\times \\sqrt{N}$
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_ret = returns.mean() - (risk_free_rate / periods)
    return (excess_ret / returns.std()) * np.sqrt(periods)

def calculate_max_drawdown(equity: pd.Series) -> float:
    """
    Computes Maximum Drawdown (MDD) from the equity curve.
    
    Formula: $MDD = \\min_t \\left( \\frac{V_t - HWM_t}{HWM_t} \\right)$
    where $HWM_t = \\max_{\\tau \\le t} V_\\tau$.
    
    Returns:
        float: Negative decimal (e.g., -0.20 for 20% DD). Returns -1.0 on bankruptcy.
    """
    if equity.empty or (equity <= 0).all():
        return -1.0 
        
    running_max = equity.cummax()
    # Numerical Stability: Replace 0 with NaN to handle bankruptcy cases gracefully
    safe_max = running_max.replace(0, np.nan)
    drawdowns = (equity - safe_max) / safe_max
    
    mdd = drawdowns.min()
    return mdd if not np.isnan(mdd) else 0.0

# ==================== FORMATTING (REFINED) ====================

def format_large_number(num: float) -> str:
    """
    Scales large values into human-readable strings (e.g., 1_000_000 -> $1.00M).
    """
    if num is None or np.isnan(num): return "$0.00"
    
    suffix = ['', 'K', 'M', 'B', 'T']
    idx = 0
    val = abs(num)
    
    while val >= 1000 and idx < len(suffix) - 1:
        val /= 1000.0
        idx += 1
    
    # Sign preservation handled implicitly via conditional multiplication
    return f"${val * (1 if num >= 0 else -1):.2f}{suffix[idx]}"

def format_bps(num: float) -> str:
    """
    Converts decimal rate to basis points notation.
    
    Example: 0.0005 -> '5.0 bps'
    """
    return f"{num * 10000:.1f} bps"