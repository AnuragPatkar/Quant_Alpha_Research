"""
Factor Analysis
===============
Tools for analyzing alpha factors and model predictions.

Features:
- Alpha decay analysis
- Factor correlation analysis
- Factor turnover calculation
- Redundancy detection

Author: [Your Name]
Last Updated: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def analyze_alpha_decay(
    predictions_df: pd.DataFrame,
    max_horizon: int = 63,
    horizons: Optional[List[int]] = None,
    pred_col: str = 'prediction',
    return_col: str = 'forward_return',
    date_col: str = 'date',
    ticker_col: str = 'ticker',
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Analyze how alpha signal decays over different holding periods.
    
    Alpha typically decays as:
    1. Information gets priced in
    2. Other traders act on similar signals
    3. Fundamental changes occur
    
    Args:
        predictions_df: DataFrame with predictions
        max_horizon: Maximum horizon to test (trading days)
        horizons: Specific horizons to test (default: [1, 5, 10, 21, 42, 63])
        pred_col: Prediction column name
        return_col: Return column name (used as base)
        date_col: Date column name
        ticker_col: Ticker column name
        price_col: Price column name (if calculating fresh returns)
        
    Returns:
        DataFrame with IC at each horizon
        
    Example:
        >>> decay = analyze_alpha_decay(predictions_df)
        >>> print(decay)
        >>> # Plot decay curve
        >>> decay.plot(x='horizon', y='rank_ic')
    """
    if horizons is None:
        horizons = [1, 5, 10, 21, 42, 63]
    
    horizons = [h for h in horizons if h <= max_horizon]
    
    results = []
    
    df = predictions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col])
    
    for horizon in horizons:
        # Calculate forward return for this horizon
        if price_col in df.columns:
            df[f'return_{horizon}d'] = df.groupby(ticker_col)[price_col].pct_change(horizon).shift(-horizon)
            return_col_h = f'return_{horizon}d'
        else:
            # Scale existing return approximately
            base_horizon = 21  # Assume base is 21-day
            df[f'return_{horizon}d'] = df[return_col] * (horizon / base_horizon)
            return_col_h = f'return_{horizon}d'
        
        # Calculate cross-sectional IC for this horizon
        def calc_ic(group):
            pred = group[pred_col].values
            ret = group[return_col_h].values
            
            mask = ~(np.isnan(pred) | np.isnan(ret))
            if mask.sum() < 5:
                return pd.Series({'ic': np.nan, 'rank_ic': np.nan})
            
            ic = np.corrcoef(pred[mask], ret[mask])[0, 1]
            rank_ic, _ = stats.spearmanr(pred[mask], ret[mask])
            
            return pd.Series({
                'ic': ic if not np.isnan(ic) else 0,
                'rank_ic': rank_ic if not np.isnan(rank_ic) else 0
            })
        
        cs_ic = df.groupby(date_col).apply(calc_ic)
        
        results.append({
            'horizon': horizon,
            'mean_ic': cs_ic['ic'].mean(),
            'mean_rank_ic': cs_ic['rank_ic'].mean(),
            'ic_std': cs_ic['ic'].std(),
            'ir': cs_ic['rank_ic'].mean() / (cs_ic['rank_ic'].std() + 1e-8),
            'pct_positive': (cs_ic['rank_ic'] > 0).mean(),
            'n_dates': len(cs_ic)
        })
    
    decay_df = pd.DataFrame(results)
    
    # Calculate half-life (horizon where IC drops to half)
    if len(decay_df) > 1:
        max_ic = decay_df['mean_rank_ic'].max()
        half_ic = max_ic / 2
        
        below_half = decay_df[decay_df['mean_rank_ic'] <= half_ic]
        if len(below_half) > 0:
            half_life = below_half['horizon'].iloc[0]
            logger.info(f"Alpha half-life: ~{half_life} days")
    
    return decay_df


def calculate_factor_correlations(
    features_df: pd.DataFrame,
    feature_names: List[str],
    date_col: str = 'date',
    method: str = 'spearman'
) -> pd.DataFrame:
    """
    Calculate average cross-sectional correlations between factors.
    
    High correlations indicate redundant factors that may not add value.
    
    Args:
        features_df: DataFrame with feature values
        feature_names: List of feature column names
        date_col: Date column name
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        Correlation matrix DataFrame
        
    Example:
        >>> corr = calculate_factor_correlations(features_df, feature_names)
        >>> # Find highly correlated pairs
        >>> high_corr = corr[corr > 0.7].stack()
    """
    # Calculate correlation within each date, then average
    correlations = []
    
    for date, group in features_df.groupby(date_col):
        if len(group) < 10:
            continue
        
        corr = group[feature_names].corr(method=method)
        correlations.append(corr)
    
    if not correlations:
        return pd.DataFrame()
    
    # Average correlations across dates
    avg_corr = pd.concat(correlations).groupby(level=0).mean()
    
    return avg_corr


def calculate_factor_turnover(
    features_df: pd.DataFrame,
    feature_names: List[str],
    date_col: str = 'date',
    ticker_col: str = 'ticker',
    top_n: int = 10
) -> pd.DataFrame:
    """
    Calculate factor turnover - how much factor rankings change over time.
    
    High turnover = high transaction costs if used directly.
    
    Args:
        features_df: DataFrame with feature values
        feature_names: List of feature column names
        date_col: Date column name
        ticker_col: Ticker column name
        top_n: Number of top stocks to track
        
    Returns:
        DataFrame with turnover statistics per factor
    """
    df = features_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    dates = sorted(df[date_col].unique())
    
    results = []
    
    for feature in feature_names:
        turnovers = []
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            prev_data = df[df[date_col] == prev_date]
            curr_data = df[df[date_col] == curr_date]
            
            if len(prev_data) < top_n or len(curr_data) < top_n:
                continue
            
            # Get top N stocks by factor value
            prev_top = set(prev_data.nlargest(top_n, feature)[ticker_col])
            curr_top = set(curr_data.nlargest(top_n, feature)[ticker_col])
            
            # Calculate turnover (1 - overlap)
            overlap = len(prev_top & curr_top) / top_n
            turnover = 1 - overlap
            turnovers.append(turnover)
        
        if turnovers:
            results.append({
                'feature': feature,
                'mean_turnover': np.mean(turnovers),
                'std_turnover': np.std(turnovers),
                'median_turnover': np.median(turnovers),
                'annual_turnover': np.mean(turnovers) * 252  # Approximate annual
            })
    
    return pd.DataFrame(results).sort_values('mean_turnover')


def analyze_factor_returns(
    features_df: pd.DataFrame,
    feature_names: List[str],
    return_col: str = 'forward_return',
    date_col: str = 'date',
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    Analyze returns by factor quantiles.
    
    For a good factor, top quantile should have higher returns than bottom.
    
    Args:
        features_df: DataFrame with features and returns
        feature_names: List of feature column names
        return_col: Return column name
        date_col: Date column name
        n_quantiles: Number of quantiles to create
        
    Returns:
        DataFrame with quantile returns per factor
    """
    results = []
    
    for feature in feature_names:
        quantile_returns = []
        
        for date, group in features_df.groupby(date_col):
            if len(group) < n_quantiles * 2:
                continue
            
            try:
                group['quantile'] = pd.qcut(
                    group[feature], 
                    n_quantiles, 
                    labels=range(1, n_quantiles + 1),
                    duplicates='drop'
                )
                
                q_ret = group.groupby('quantile')[return_col].mean()
                quantile_returns.append(q_ret)
            except:
                continue
        
        if quantile_returns:
            avg_returns = pd.concat(quantile_returns, axis=1).mean(axis=1)
            
            results.append({
                'feature': feature,
                'q1_return': avg_returns.get(1, np.nan),
                'q5_return': avg_returns.get(n_quantiles, np.nan),
                'spread': avg_returns.get(n_quantiles, 0) - avg_returns.get(1, 0),
                'monotonic': _check_monotonicity(avg_returns),
                'n_dates': len(quantile_returns)
            })
    
    return pd.DataFrame(results).sort_values('spread', ascending=False)


def _check_monotonicity(series: pd.Series) -> float:
    """Check if series is monotonically increasing (returns 0-1 score)."""
    if len(series) < 2:
        return 0.5
    
    values = series.sort_index().values
    increasing = sum(values[i] <= values[i+1] for i in range(len(values)-1))
    
    return increasing / (len(values) - 1)


def get_redundant_factors(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7
) -> List[Tuple[str, str, float]]:
    """
    Identify redundant factor pairs based on correlation.
    
    Args:
        corr_matrix: Correlation matrix from calculate_factor_correlations
        threshold: Correlation threshold for redundancy
        
    Returns:
        List of (factor1, factor2, correlation) tuples
    """
    redundant = []
    
    for i, col1 in enumerate(corr_matrix.columns):
        for col2 in corr_matrix.columns[i+1:]:
            corr = corr_matrix.loc[col1, col2]
            if abs(corr) >= threshold:
                redundant.append((col1, col2, corr))
    
    return sorted(redundant, key=lambda x: abs(x[2]), reverse=True)


def calculate_factor_ic_by_period(
    predictions_df: pd.DataFrame,
    pred_col: str = 'prediction',
    return_col: str = 'forward_return',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Calculate IC broken down by time period (year, quarter, month).
    
    Useful for identifying when the model works best/worst.
    
    Args:
        predictions_df: DataFrame with predictions and returns
        pred_col: Prediction column name
        return_col: Return column name
        date_col: Date column name
        
    Returns:
        DataFrame with IC by period
    """
    df = predictions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['quarter'] = df[date_col].dt.quarter
    df['month'] = df[date_col].dt.month
    
    results = []
    
    # By year
    for year, group in df.groupby('year'):
        ic = _calc_pooled_ic(group, pred_col, return_col)
        results.append({
            'period': f'Year {year}',
            'period_type': 'year',
            'ic': ic['ic'],
            'rank_ic': ic['rank_ic'],
            'n_samples': len(group)
        })
    
    # By quarter
    for (year, quarter), group in df.groupby(['year', 'quarter']):
        ic = _calc_pooled_ic(group, pred_col, return_col)
        results.append({
            'period': f'{year} Q{quarter}',
            'period_type': 'quarter',
            'ic': ic['ic'],
            'rank_ic': ic['rank_ic'],
            'n_samples': len(group)
        })
    
    return pd.DataFrame(results)


def _calc_pooled_ic(df, pred_col, return_col):
    """Calculate pooled IC for a DataFrame subset."""
    pred = df[pred_col].values
    ret = df[return_col].values
    
    mask = ~(np.isnan(pred) | np.isnan(ret))
    if mask.sum() < 5:
        return {'ic': 0, 'rank_ic': 0}
    
    ic = np.corrcoef(pred[mask], ret[mask])[0, 1]
    rank_ic, _ = stats.spearmanr(pred[mask], ret[mask])
    
    return {
        'ic': ic if not np.isnan(ic) else 0,
        'rank_ic': rank_ic if not np.isnan(rank_ic) else 0
    }