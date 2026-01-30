"""
Market Regime Analysis
======================
Analyze model performance across different market regimes.

Features:
- Regime identification (bull/bear/sideways)
- Volatility regime detection
- Performance breakdown by regime

Author: [Your Name]
Last Updated: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    CRISIS = "crisis"


def identify_market_regimes(
    market_returns: pd.Series,
    volatility_window: int = 21,
    trend_window: int = 63,
    vol_threshold_percentile: int = 75,
    trend_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Identify market regimes based on returns and volatility.
    
    Args:
        market_returns: Series of market (benchmark) returns
        volatility_window: Window for volatility calculation
        trend_window: Window for trend calculation
        vol_threshold_percentile: Percentile for high/low vol split
        trend_threshold: Annual return threshold for bull/bear
        
    Returns:
        DataFrame with regime labels
    """
    df = pd.DataFrame({'return': market_returns})
    df.index = pd.to_datetime(df.index)
    
    # Calculate rolling volatility (annualized)
    df['volatility'] = df['return'].rolling(volatility_window).std() * np.sqrt(252)
    
    # Calculate rolling return (annualized)
    df['rolling_return'] = df['return'].rolling(trend_window).mean() * 252
    
    # Volatility regime
    vol_threshold = df['volatility'].quantile(vol_threshold_percentile / 100)
    df['vol_regime'] = np.where(
        df['volatility'] > vol_threshold,
        MarketRegime.HIGH_VOL.value,
        MarketRegime.LOW_VOL.value
    )
    
    # Trend regime
    df['trend_regime'] = MarketRegime.SIDEWAYS.value
    df.loc[df['rolling_return'] > trend_threshold, 'trend_regime'] = MarketRegime.BULL.value
    df.loc[df['rolling_return'] < -trend_threshold, 'trend_regime'] = MarketRegime.BEAR.value
    
    # Crisis detection (high vol + negative returns)
    crisis_mask = (df['vol_regime'] == MarketRegime.HIGH_VOL.value) & \
                  (df['trend_regime'] == MarketRegime.BEAR.value)
    df.loc[crisis_mask, 'trend_regime'] = MarketRegime.CRISIS.value
    
    # Combined regime label
    df['regime'] = df['trend_regime'] + '_' + df['vol_regime']
    
    return df


def analyze_regime_performance(
    predictions_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    pred_col: str = 'prediction',
    return_col: str = 'forward_return',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Analyze model performance by market regime.
    
    Args:
        predictions_df: DataFrame with predictions
        regime_df: DataFrame with regime labels (from identify_market_regimes)
        pred_col: Prediction column name
        return_col: Return column name
        date_col: Date column name
        
    Returns:
        DataFrame with performance metrics per regime
    """
    df = predictions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Merge with regime data
    regime_df.index = pd.to_datetime(regime_df.index)
    df = df.merge(
        regime_df[['regime', 'trend_regime', 'vol_regime']], 
        left_on=date_col, 
        right_index=True,
        how='left'
    )
    
    results = []
    
    # Analyze by trend regime
    for regime in df['trend_regime'].dropna().unique():
        regime_data = df[df['trend_regime'] == regime]
        
        if len(regime_data) < 50:
            continue
        
        metrics = _calculate_regime_metrics(regime_data, pred_col, return_col)
        metrics['regime'] = regime
        metrics['regime_type'] = 'trend'
        metrics['n_observations'] = len(regime_data)
        results.append(metrics)
    
    # Analyze by volatility regime
    for regime in df['vol_regime'].dropna().unique():
        regime_data = df[df['vol_regime'] == regime]
        
        if len(regime_data) < 50:
            continue
        
        metrics = _calculate_regime_metrics(regime_data, pred_col, return_col)
        metrics['regime'] = regime
        metrics['regime_type'] = 'volatility'
        metrics['n_observations'] = len(regime_data)
        results.append(metrics)
    
    return pd.DataFrame(results)


def _calculate_regime_metrics(
    df: pd.DataFrame,
    pred_col: str,
    return_col: str
) -> Dict[str, float]:
    """Calculate metrics for a regime subset."""
    from scipy import stats
    
    pred = df[pred_col].values
    ret = df[return_col].values
    
    mask = ~(np.isnan(pred) | np.isnan(ret))
    
    if mask.sum() < 10:
        return {'ic': 0, 'rank_ic': 0, 'hit_rate': 0.5}
    
    pred_clean = pred[mask]
    ret_clean = ret[mask]
    
    ic = np.corrcoef(pred_clean, ret_clean)[0, 1]
    rank_ic, _ = stats.spearmanr(pred_clean, ret_clean)
    hit_rate = np.mean(np.sign(pred_clean) == np.sign(ret_clean))
    
    return {
        'ic': float(ic) if not np.isnan(ic) else 0,
        'rank_ic': float(rank_ic) if not np.isnan(rank_ic) else 0,
        'hit_rate': float(hit_rate)
    }


def calculate_regime_metrics(
    returns: pd.Series,
    regime_labels: pd.Series
) -> pd.DataFrame:
    """
    Calculate return statistics by regime.
    
    Args:
        returns: Strategy returns
        regime_labels: Regime labels aligned with returns
        
    Returns:
        DataFrame with statistics per regime
    """
    df = pd.DataFrame({
        'return': returns,
        'regime': regime_labels
    })
    
    results = []
    
    for regime, group in df.groupby('regime'):
        rets = group['return'].dropna()
        
        if len(rets) < 5:
            continue
        
        results.append({
            'regime': regime,
            'mean_return': rets.mean() * 252,  # Annualized
            'volatility': rets.std() * np.sqrt(252),
            'sharpe': (rets.mean() / (rets.std() + 1e-10)) * np.sqrt(252),
            'max_drawdown': _calculate_max_drawdown(rets),
            'win_rate': (rets > 0).mean(),
            'n_days': len(rets),
            'pct_of_time': len(rets) / len(df) * 100
        })
    
    return pd.DataFrame(results)


def _calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    return float(drawdowns.min())


def detect_regime_changes(
    regime_df: pd.DataFrame,
    regime_col: str = 'trend_regime'
) -> pd.DataFrame:
    """
    Detect regime change points.
    
    Args:
        regime_df: DataFrame with regime labels
        regime_col: Column with regime labels
        
    Returns:
        DataFrame with regime change dates
    """
    df = regime_df.copy()
    df['prev_regime'] = df[regime_col].shift(1)
    df['regime_change'] = df[regime_col] != df['prev_regime']
    
    changes = df[df['regime_change']].copy()
    changes['from_regime'] = changes['prev_regime']
    changes['to_regime'] = changes[regime_col]
    
    return changes[['from_regime', 'to_regime']].dropna()