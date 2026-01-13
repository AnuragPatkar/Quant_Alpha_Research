"""
Factor Registry
===============
Central registry for all alpha factors.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings


class FactorRegistry:
    """
    Central registry for all factors.
    
    Computes all features in one pipeline.
    """
    
    def __init__(self):
        """Initialize with all factors."""
        self.feature_names = []
        self.config = settings.features
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for all stocks.
        
        Args:
            df: Raw OHLCV DataFrame with columns [date, ticker, open, high, low, close, volume]
            
        Returns:
            DataFrame with all features added
        """
        print("\n" + "="*60)
        print("🔧 FEATURE ENGINEERING")
        print("="*60)
        
        self.feature_names = []
        results = []
        
        tickers = df['ticker'].unique()
        print(f"   Processing {len(tickers)} stocks...")
        
        for ticker in tqdm(tickers, desc="   Computing features"):
            # Get single stock data
            stock_df = df[df['ticker'] == ticker].copy()
            stock_df = stock_df.sort_values('date').reset_index(drop=True)
            
            # Compute all features
            stock_df = self._add_momentum_features(stock_df)
            stock_df = self._add_mean_reversion_features(stock_df)
            stock_df = self._add_volatility_features(stock_df)
            stock_df = self._add_volume_features(stock_df)
            stock_df = self._add_target(stock_df)
            
            results.append(stock_df)
        
        # Combine all stocks
        result = pd.concat(results, ignore_index=True)
        
        # Drop NaN rows
        initial = len(result)
        result = result.dropna()
        dropped = initial - len(result)
        
        # Remove duplicates from feature names
        self.feature_names = list(dict.fromkeys(self.feature_names))
        
        # Print summary
        print(f"\n   ✅ Created {len(self.feature_names)} features")
        print(f"   📉 Dropped {dropped:,} rows (NaN from rolling windows)")
        print(f"   📊 Final dataset: {len(result):,} rows")
        
        self._print_feature_summary()
        
        return result
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based features."""
        close = df['close']
        
        # Calculate returns over different time periods
        # Using standard windows: 1w, 2w, 1m, 3m, 6m
        for window in self.config.momentum_windows:
            col = f'mom_{window}d'
            df[col] = close.pct_change(window)
            self._add_feature(col)
        
        # Rate of change - alternative momentum measure
        df['roc_21d'] = (close - close.shift(21)) / (close.shift(21) + 1e-10)
        self._add_feature('roc_21d')
        
        # EMA-based momentum (similar to MACD concept)
        # Using 12 and 26 day EMAs as standard
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['ema_momentum'] = (ema_12 - ema_26) / (close + 1e-10)
        self._add_feature('ema_momentum')
        
        return df
    
    def _add_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators."""
        close = df['close']
        
        # RSI
        for window in self.config.rsi_windows:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / (loss + 1e-10)
            col = f'rsi_{window}d'
            df[col] = 100 - (100 / (1 + rs))
            self._add_feature(col)
        
        # Distance from moving averages
        for window in self.config.ma_windows:
            ma = close.rolling(window).mean()
            col = f'dist_ma_{window}d'
            df[col] = (close - ma) / (ma + 1e-10)
            self._add_feature(col)
        
        # Z-score
        for window in [10, 21]:
            mean = close.rolling(window).mean()
            std = close.rolling(window).std()
            col = f'zscore_{window}d'
            df[col] = (close - mean) / (std + 1e-10)
            self._add_feature(col)
        
        # Bollinger Band position
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        upper = ma_20 + 2 * std_20
        lower = ma_20 - 2 * std_20
        df['bb_position'] = (close - lower) / (upper - lower + 1e-10)
        self._add_feature('bb_position')
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        close = df['close']
        high = df['high']
        low = df['low']
        returns = close.pct_change()
        
        # Historical volatility
        for window in self.config.volatility_windows:
            col = f'volatility_{window}d'
            df[col] = returns.rolling(window).std() * np.sqrt(252)
            self._add_feature(col)
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14d'] = tr.rolling(14).mean() / (close + 1e-10)
        self._add_feature('atr_14d')
        
        # Volatility ratio
        if 'volatility_10d' in df.columns and 'volatility_63d' in df.columns:
            df['vol_ratio'] = df['volatility_10d'] / (df['volatility_63d'] + 1e-10)
            self._add_feature('vol_ratio')
        
        # Skewness
        df['skewness_21d'] = returns.rolling(21).skew()
        self._add_feature('skewness_21d')
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume features."""
        volume = df['volume']
        close = df['close']
        returns = close.pct_change()
        
        # Volume Z-score
        for window in [10, 21]:
            vol_mean = volume.rolling(window).mean()
            vol_std = volume.rolling(window).std()
            col = f'volume_zscore_{window}d'
            df[col] = (volume - vol_mean) / (vol_std + 1e-10)
            self._add_feature(col)
        
        # Relative volume
        df['relative_volume'] = volume / (volume.rolling(21).mean() + 1e-10)
        self._add_feature('relative_volume')
        
        # Price-volume correlation
        df['pv_corr_21d'] = returns.rolling(21).corr(volume.pct_change())
        self._add_feature('pv_corr_21d')
        
        # Amihud illiquidity
        abs_ret = returns.abs()
        dollar_vol = close * volume
        daily_illiq = abs_ret / (dollar_vol + 1e-10)
        df['amihud_21d'] = np.log1p(daily_illiq * 1e6).rolling(21).mean()
        self._add_feature('amihud_21d')
        
        return df
    
    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add forward return target."""
        n_days = self.config.forward_return_days
        df['forward_return'] = df['close'].pct_change(n_days).shift(-n_days)
        return df
    
    def _add_feature(self, name: str):
        """Add feature name to list."""
        if name not in self.feature_names:
            self.feature_names.append(name)
    
    def _print_feature_summary(self):
        """Print feature summary by category."""
        categories = {
            'Momentum': [f for f in self.feature_names if 'mom' in f or 'roc' in f or 'ema' in f],
            'Mean Reversion': [f for f in self.feature_names if 'rsi' in f or 'dist' in f or 'zscore' in f or 'bb' in f],
            'Volatility': [f for f in self.feature_names if 'volatility' in f or 'atr' in f or 'vol_ratio' in f or 'skew' in f],
            'Volume': [f for f in self.feature_names if 'volume' in f or 'pv_' in f or 'amihud' in f or 'relative' in f]
        }
        
        print(f"\n   📋 Features by Category:")
        for cat, features in categories.items():
            if features:
                print(f"      {cat}: {len(features)} features")
                for f in features[:3]:
                    print(f"         • {f}")
                if len(features) > 3:
                    print(f"         • ... ({len(features) - 3} more)")
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()


def compute_all_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function to compute all features.
    
    Args:
        df: Raw OHLCV DataFrame
        
    Returns:
        Tuple of (features_df, feature_names)
    """
    registry = FactorRegistry()
    features_df = registry.compute_features(df)
    return features_df, registry.get_feature_names()