"""
Backtesting Utilities - Optimized & Hardened
Final Version with performance fixes, safety guards, and core dependencies.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ==================== RETURN CALCULATIONS ====================

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """Calculates periodic returns. Essential for engine/metrics pipeline."""
    if method == 'log':
        return np.log(prices / prices.shift(1))
    return prices.pct_change()

def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calculates equity growth (1.0 based). Required for equity curve generation."""
    return (1 + returns.fillna(0)).cumprod()

# ==================== DATE OPERATIONS (OPTIMIZED) ====================

def align_dates(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Panel-Safe Date Alignment using Set Intersection.
    OPTIMIZATION: Date conversion (pd.to_datetime) is performed ONLY ONCE per 
    dataframe to eliminate bottlenecks in large panel datasets.
    """
    if df1.empty or df2.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Convert once and reuse to avoid expensive repeated parsing
    d1_temp = pd.to_datetime(df1['date'])
    d2_temp = pd.to_datetime(df2['date'])
    
    # Intersection of unique dates prevents Cartesian explosion
    common_dates = sorted(list(set(d1_temp).intersection(set(d2_temp))))
    
    if not common_dates:
        logger.warning("No common dates found between datasets.")
        return pd.DataFrame(), pd.DataFrame()

    # Vectorized filtering using pre-converted series
    df1_aligned = df1[d1_temp.isin(common_dates)].copy()
    df2_aligned = df2[d2_temp.isin(common_dates)].copy()
    
    return df1_aligned, df2_aligned

# ==================== DATA VALIDATION (HARDENED) ====================

def validate_backtest_data(predictions: pd.DataFrame, prices: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Hardened validation ensuring numeric integrity and signal consistency.
    Essential for preventing crashes during matrix operations (pivot/dot).
    """
    errors = []
    
    # 1. Column Existence Check
    pred_req = {'date', 'ticker', 'prediction'}
    price_req = {'date', 'ticker', 'close'}
    
    if not pred_req.issubset(predictions.columns):
        errors.append(f"Predictions missing: {pred_req - set(predictions.columns)}")
    if not price_req.issubset(prices.columns):
        errors.append(f"Prices missing: {price_req - set(prices.columns)}")
    
    if errors: return False, errors

    # 2. Numeric Integrity Check (Prevents string/object related math errors)
    if not pd.api.types.is_numeric_dtype(predictions['prediction']):
        errors.append("Prediction column must be numeric.")
    
    if not pd.api.types.is_numeric_dtype(prices['close']):
        errors.append("Price close column must be numeric.")

    # 3. Duplicate Check (Ensures unique index for pivoting)
    if predictions.duplicated(subset=['date', 'ticker']).any():
        errors.append("Duplicate (date, ticker) pairs in predictions found.")

    return len(errors) == 0, errors

# ==================== PERFORMANCE HELPERS ====================

def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02, periods: int = 252) -> float:
    """Annualized Sharpe Ratio calculation."""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_ret = returns.mean() - (risk_free_rate / periods)
    return (excess_ret / returns.std()) * np.sqrt(periods)

def calculate_max_drawdown(equity: pd.Series) -> float:
    """
    Calculates Max Drawdown with Division-by-Zero protection.
    Returns: -1.0 for total bankruptcy, otherwise decimal percentage.
    """
    if equity.empty or (equity <= 0).all():
        return -1.0 
        
    running_max = equity.cummax()
    # Replace 0 with NaN to avoid division-by-zero, then execute calculation
    safe_max = running_max.replace(0, np.nan)
    drawdowns = (equity - safe_max) / safe_max
    
    mdd = drawdowns.min()
    return mdd if not np.isnan(mdd) else 0.0

# ==================== FORMATTING (REFINED) ====================

def format_large_number(num: float) -> str:
    """Efficient loop-based scaling for large financial figures (K, M, B, T)."""
    if num is None or np.isnan(num): return "$0.00"
    
    suffix = ['', 'K', 'M', 'B', 'T']
    idx = 0
    val = abs(num)
    
    while val >= 1000 and idx < len(suffix) - 1:
        val /= 1000.0
        idx += 1
    
    # Preserve sign while formatting
    return f"${val * (1 if num >= 0 else -1):.2f}{suffix[idx]}"

def format_bps(num: float) -> str:
    """Converts decimal to basis points (e.g., 0.0005 -> 5.0 bps)."""
    return f"{num * 10000:.1f} bps"