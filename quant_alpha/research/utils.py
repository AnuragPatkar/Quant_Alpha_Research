import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def prepare_factor_data(data: pd.DataFrame, factor_col: str, forward_return_col: str = 'raw_ret_5d'):
    """
    Prepares a clean DataFrame with MultiIndex (date, ticker) for analysis.
    """
    required = ['date', 'ticker', factor_col, forward_return_col]
    
    # Check columns
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
        
    df = data[required].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Drop NaNs
    original_len = len(df)
    df = df.dropna()
    dropped = original_len - len(df)
    if dropped > 0:
        logger.debug(f"Dropped {dropped} rows with NaNs in factor or target.")
        
    return df.set_index(['date', 'ticker']).sort_index()

def get_forward_returns_columns(data: pd.DataFrame):
    """Detects forward return columns in the dataset."""
    return [c for c in data.columns if 'ret' in c and ('fwd' in c or 'future' in c or 'raw' in c)]

def plot_distribution(series: pd.Series, title: str = "Distribution", save_path=None):
    """Plots histogram with KDE."""
    plt.figure(figsize=(10, 6))
    sns.histplot(series, kde=True, bins=50)
    plt.title(title)
    plt.axvline(series.mean(), color='r', linestyle='--', label=f'Mean: {series.mean():.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def winsorize_series(s: pd.Series, limits=(0.01, 0.01)):
    """Winsorizes a pandas Series."""
    from scipy.stats.mstats import winsorize
    return pd.Series(winsorize(s, limits=limits), index=s.index)

def standardize_series(s: pd.Series):
    """Z-score standardization."""
    return (s - s.mean()) / (s.std() + 1e-8)
