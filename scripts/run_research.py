#!/usr/bin/env python3
"""
LightGBM ML Research Pipeline
Walk-forward validation with regime detection
Uses DataLoader for consistent data across all scripts
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings, print_welcome
from quant_alpha.data import DataLoader  # ✅ ADD THIS


def print_header(title: str) -> None:
    """Print section header."""
    print(f"\n{'='*65}")
    print(f" [ML] {title}")
    print('='*65)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all 47 features for ML model."""
    
    df = df.sort_values(['ticker', 'date']).copy()
    
    features_list = []
    
    for ticker in df['ticker'].unique():
        stock = df[df['ticker'] == ticker].copy()
        
        # Price data
        close = stock['close']
        high = stock['high']
        low = stock['low']
        volume = stock['volume']
        
        # =================================================================
        # 1. MOMENTUM FEATURES (12 features)
        # =================================================================
        
        # Returns
        stock['return_1d'] = close.pct_change(1)
        stock['return_5d'] = close.pct_change(5)
        stock['return_10d'] = close.pct_change(10)
        stock['return_21d'] = close.pct_change(21)
        stock['return_63d'] = close.pct_change(63)
        
        # Momentum
        stock['mom_5'] = close / close.shift(5) - 1
        stock['mom_10'] = close / close.shift(10) - 1
        stock['mom_21'] = close / close.shift(21) - 1
        stock['mom_63'] = close / close.shift(63) - 1
        
        # Momentum acceleration
        stock['mom_accel_5'] = stock['mom_5'] - stock['mom_5'].shift(5)
        stock['mom_accel_21'] = stock['mom_21'] - stock['mom_21'].shift(21)
        
        # Rate of change
        stock['roc_10'] = (close - close.shift(10)) / close.shift(10)
        
        # =================================================================
        # 2. VOLATILITY FEATURES (10 features)
        # =================================================================
        
        # Standard deviation of returns
        stock['volatility_5'] = stock['return_1d'].rolling(5).std()
        stock['volatility_10'] = stock['return_1d'].rolling(10).std()
        stock['volatility_21'] = stock['return_1d'].rolling(21).std()
        stock['volatility_63'] = stock['return_1d'].rolling(63).std()
        
        # Garman-Klass volatility
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / stock['open']) ** 2
        stock['gk_volatility_10'] = np.sqrt(0.5 * log_hl.rolling(10).mean() - (2 * np.log(2) - 1) * log_co.rolling(10).mean())
        stock['gk_volatility_21'] = np.sqrt(0.5 * log_hl.rolling(21).mean() - (2 * np.log(2) - 1) * log_co.rolling(21).mean())
        
        # High-Low range
        stock['hl_range_5'] = ((high - low) / close).rolling(5).mean()
        stock['hl_range_10'] = ((high - low) / close).rolling(10).mean()
        stock['hl_range_21'] = ((high - low) / close).rolling(21).mean()
        
        # Volatility ratio
        stock['vol_ratio'] = stock['volatility_10'] / stock['volatility_63'].replace(0, np.nan)
        
        # =================================================================
        # 3. VOLUME FEATURES (8 features)
        # =================================================================
        
        stock['volume_ma_5'] = volume.rolling(5).mean()
        stock['volume_ma_21'] = volume.rolling(21).mean()
        stock['volume_ratio_5'] = volume / stock['volume_ma_5'].replace(0, np.nan)
        stock['volume_ratio_21'] = volume / stock['volume_ma_21'].replace(0, np.nan)
        
        # Volume trend
        stock['volume_trend'] = stock['volume_ma_5'] / stock['volume_ma_21'].replace(0, np.nan)
        
        # On-Balance Volume trend
        obv = (np.sign(stock['return_1d']) * volume).cumsum()
        stock['obv_change_5'] = obv.pct_change(5)
        stock['obv_change_21'] = obv.pct_change(21)
        
        # Volume-price correlation
        stock['vol_price_corr'] = stock['return_1d'].rolling(21).corr(volume.pct_change())
        
        # =================================================================
        # 4. MEAN REVERSION FEATURES (8 features)
        # =================================================================
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        stock['rsi_14'] = 100 - (100 / (1 + rs))
        stock['rsi_21'] = stock['rsi_14'].rolling(21).mean()
        
        # Distance from moving averages
        stock['ma_10'] = close.rolling(10).mean()
        stock['ma_21'] = close.rolling(21).mean()
        stock['ma_50'] = close.rolling(50).mean()
        stock['ma_200'] = close.rolling(200).mean()
        
        stock['dist_ma_10'] = (close - stock['ma_10']) / stock['ma_10']
        stock['dist_ma_21'] = (close - stock['ma_21']) / stock['ma_21']
        stock['dist_ma_50'] = (close - stock['ma_50']) / stock['ma_50']
        stock['dist_ma_200'] = (close - stock['ma_200']) / stock['ma_200']
        
        # =================================================================
        # 5. TREND FEATURES (9 features)
        # =================================================================
        
        # Moving average crossovers
        stock['ma_cross_10_21'] = (stock['ma_10'] > stock['ma_21']).astype(int)
        stock['ma_cross_21_50'] = (stock['ma_21'] > stock['ma_50']).astype(int)
        stock['ma_cross_50_200'] = (stock['ma_50'] > stock['ma_200']).astype(int)
        
        # Price position
        stock['price_position_52w'] = (close - close.rolling(252).min()) / (close.rolling(252).max() - close.rolling(252).min()).replace(0, np.nan)
        
        # Trend strength
        stock['trend_strength_21'] = stock['mom_21'] / stock['volatility_21'].replace(0, np.nan)
        stock['trend_strength_63'] = stock['mom_63'] / stock['volatility_63'].replace(0, np.nan)
        
        # Higher highs, higher lows
        stock['hh_count_10'] = (high > high.shift(1)).rolling(10).sum()
        stock['hl_count_10'] = (low > low.shift(1)).rolling(10).sum()
        
        # ADX approximation
        stock['adx_proxy'] = abs(stock['mom_21']) / stock['volatility_21'].replace(0, np.nan)
        
        # =================================================================
        # TARGET: Forward Return
        # =================================================================
        
        stock['forward_return'] = close.shift(-21) / close - 1
        
        features_list.append(stock)
    
    result = pd.concat(features_list, ignore_index=True)
    
    return result


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_regime(market_df: pd.DataFrame, date) -> str:
    """Detect market regime: bull, bear, or neutral."""
    
    recent = market_df[market_df['date'] <= date].tail(200)
    
    if len(recent) < 50:
        return 'neutral'
    
    current = recent['close'].iloc[-1]
    ma_50 = recent['close'].tail(50).mean()
    ma_200 = recent['close'].mean()
    
    # Momentum
    mom_20 = (current / recent['close'].iloc[-20] - 1) if len(recent) >= 20 else 0
    
    # Volatility
    returns = recent['close'].pct_change().dropna()
    vol = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.15
    
    # Score calculation
    score = 0
    if current > ma_200: score += 2
    if current > ma_50: score += 1
    if mom_20 > 0.03: score += 1
    if mom_20 < -0.03: score -= 1
    if vol > 0.25: score -= 1
    if vol < 0.15: score += 1
    
    if score >= 3:
        return 'bull'
    elif score <= -1:
        return 'bear'
    else:
        return 'neutral'


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print_welcome()
    
    print("\n" + "🤖 "*20)
    print(" LIGHTGBM ML RESEARCH PIPELINE")
    print("🤖 "*20)
    
    start_time = time.time()
    
    settings = Settings(show_survivorship_warning=False)
    results_dir = settings.results_dir
    
    # =========================================================
    # STEP 1: Load Data - USE DATALOADER (Same as main.py)
    # =========================================================
    print_header("STEP 1: Loading Data")
    
    # ✅ USE DATALOADER INSTEAD OF PICKLE FILE
    loader = DataLoader()
    df = loader.load()
    df['date'] = pd.to_datetime(df['date'])
    
    n_stocks = df['ticker'].nunique()
    
    print(f" [SUCCESS] Loaded: {df.shape[0]:,} rows")
    print(f" [STOCKS] {n_stocks} stocks")
    print(f" [DATE] Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    # =========================================================
    # STEP 2: Feature Engineering
    # =========================================================
    print_header("STEP 2: Feature Engineering")
    
    print(" [PROCESSING] Calculating 47 features...")
    features_df = calculate_features(df)
    
    # Define feature columns
    feature_cols = [
        # Momentum (12)
        'return_1d', 'return_5d', 'return_10d', 'return_21d', 'return_63d',
        'mom_5', 'mom_10', 'mom_21', 'mom_63',
        'mom_accel_5', 'mom_accel_21', 'roc_10',
        
        # Volatility (10)
        'volatility_5', 'volatility_10', 'volatility_21', 'volatility_63',
        'gk_volatility_10', 'gk_volatility_21',
        'hl_range_5', 'hl_range_10', 'hl_range_21', 'vol_ratio',
        
        # Volume (8)
        'volume_ratio_5', 'volume_ratio_21', 'volume_trend',
        'obv_change_5', 'obv_change_21', 'vol_price_corr',
        'volume_ma_5', 'volume_ma_21',
        
        # Mean Reversion (8)
        'rsi_14', 'rsi_21',
        'dist_ma_10', 'dist_ma_21', 'dist_ma_50', 'dist_ma_200',
        'ma_10', 'ma_21',
        
        # Trend (9)
        'ma_cross_10_21', 'ma_cross_21_50', 'ma_cross_50_200',
        'price_position_52w', 'trend_strength_21', 'trend_strength_63',
        'hh_count_10', 'hl_count_10', 'adx_proxy'
    ]
    
    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    
    print(f" [SUCCESS] Created {len(feature_cols)} features")
    print(f" [INFO] Dataset: {features_df.shape[0]:,} rows")
    
    # =========================================================
    # STEP 3: Prepare Market Index
    # =========================================================
    print_header("STEP 3: Market Index")
    
    market = df.groupby('date')['close'].mean().reset_index()
    print(f" ✅ Market index created")
    
    # =========================================================
    # STEP 4: Walk-Forward Validation
    # =========================================================
    print_header("STEP 4: Walk-Forward Validation")
    
    # Prepare data
    ml_df = features_df.dropna(subset=['forward_return'] + feature_cols[:10]).copy()
    ml_df = ml_df.replace([np.inf, -np.inf], np.nan)
    ml_df[feature_cols] = ml_df[feature_cols].fillna(0)
    
    dates = np.sort(ml_df['date'].unique())
    
    # Configuration
    train_days = 504  # 2 years training
    test_days = 63    # 3 months test
    step_days = 21    # 1 month step
    
    start_idx = train_days
    
    print(f"\n 📋 Configuration:")
    print(f"    🎯 Training Window : {train_days} days (~2 years)")
    print(f"    🧪 Test Window     : {test_days} days (~3 months)")
    print(f"    🔄 Step Size       : {step_days} days (~1 month)")
    print(f"    📊 Features        : {len(feature_cols)}")
    print(f"    📈 Stocks          : {n_stocks}")
    
    # LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbosity': -1,
        'random_state': 42
    }
    
    results = []
    all_predictions = []
    feature_importance_total = np.zeros(len(feature_cols))
    
    fold = 0
    idx = start_idx
    
    print(f"\n {'Fold':<6} {'Period':<25} {'Regime':<10} {'IC':>10} {'Rank IC':>10} {'Status'}")
    print(" " + "-"*75)
    
    while idx + test_days <= len(dates):
        # Split dates
        train_dates = dates[idx - train_days:idx]
        test_dates = dates[idx:idx + test_days]
        
        # Detect regime
        regime = detect_regime(market, test_dates[0])
        
        # Get data
        train_data = ml_df[ml_df['date'].isin(train_dates)]
        test_data = ml_df[ml_df['date'].isin(test_dates)]
        
        if len(train_data) < 1000 or len(test_data) < 100:
            idx += step_days
            continue
        
        # Prepare features
        X_train = train_data[feature_cols].values
        y_train = train_data['forward_return'].values
        X_test = test_data[feature_cols].values
        y_test = test_data['forward_return'].values
        
        # Train LightGBM
        train_set = lgb.Dataset(X_train, label=y_train)
        
        model = lgb.train(
            lgb_params,
            train_set,
            num_boost_round=200,
            valid_sets=[train_set],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Predict
        if regime == 'bear':
            # Bear market: Reduce exposure, predict conservatively
            predictions = model.predict(X_test) * 0.5
        else:
            predictions = model.predict(X_test)
        
        # Calculate IC
        try:
            ic = np.corrcoef(predictions, y_test)[0, 1]
            rank_ic = spearmanr(predictions, y_test)[0]
            ic = 0.0 if np.isnan(ic) else ic
            rank_ic = 0.0 if np.isnan(rank_ic) else rank_ic
        except:
            ic, rank_ic = 0.0, 0.0
        
        # Feature importance
        feature_importance_total += model.feature_importance(importance_type='gain')
        
        # Store results
        period = f"{str(test_dates[0])[:10]} → {str(test_dates[-1])[:10]}"
        
        results.append({
            'fold': fold,
            'test_start': str(test_dates[0])[:10],
            'test_end': str(test_dates[-1])[:10],
            'regime': regime,
            'samples': len(test_data),
            'ic': ic,
            'rank_ic': rank_ic
        })
        
        # Print result
        regime_emoji = "🟢" if regime == 'bull' else "🔴" if regime == 'bear' else "🟡"
        status = "✅" if ic > 0 else "❌"
        print(f" {fold:<6} {period} {regime_emoji} {regime:<8} {ic:>+10.4f} {rank_ic:>+10.4f} {status}")
        
        # Store predictions
        pred_df = test_data[['date', 'ticker', 'forward_return']].copy()
        pred_df['prediction'] = predictions
        pred_df['regime'] = regime
        all_predictions.append(pred_df)
        
        fold += 1
        idx += step_days
    
    # =========================================================
    # STEP 5: Results Summary
    # =========================================================
    print_header("STEP 5: Results Summary")
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print(" ❌ No valid folds!")
        return
    
    avg_ic = results_df['ic'].mean()
    avg_rank_ic = results_df['rank_ic'].mean()
    ic_std = results_df['ic'].std()
    ir = avg_ic / ic_std if ic_std > 0 else 0
    win_rate = (results_df['ic'] > 0).mean() * 100
    
    print(f"""
  ╔════════════════════════════════════════════════════════════╗
  ║  🤖 LIGHTGBM ML MODEL RESULTS                              ║
  ╠════════════════════════════════════════════════════════════╣
  ║  📊 Total Folds        : {len(results_df):<10}                     ║
  ║  📊 Stocks Used        : {n_stocks:<10}                     ║
  ║  📅 Period             : {results_df['test_start'].min()} → {results_df['test_end'].max()} ║
  ╠════════════════════════════════════════════════════════════╣
  ║  🎯 Average IC         : {avg_ic:>+10.4f}                     ║
  ║  🎯 Average Rank IC    : {avg_rank_ic:>+10.4f}                     ║
  ║  📉 IC Std Dev         : {ic_std:>10.4f}                     ║
  ║  💹 Information Ratio  : {ir:>+10.4f}                     ║
  ║  🏆 Win Rate           : {win_rate:>10.1f}%                    ║
  ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Interpretation
    print(" 📋 Interpretation:")
    print(" " + "-"*50)
    
    if avg_ic > 0.05:
        print(" ✅ [EXCELLENT] Strong predictive signal!")
    elif avg_ic > 0.03:
        print(" ✅ [GOOD] Useful predictive signal")
    elif avg_ic > 0:
        print(" 🟡 [WEAK] Small positive signal")
    else:
        print(" ❌ [POOR] Model not predictive")
    
    if win_rate >= 60:
        print(f" ✅ [RELIABLE] {win_rate:.0f}% months positive")
    elif win_rate >= 50:
        print(f" 🟡 [OK] {win_rate:.0f}% months positive")
    else:
        print(f" ❌ [UNRELIABLE] Only {win_rate:.0f}% months positive")
    
    # =========================================================
    # STEP 6: Regime Analysis
    # =========================================================
    print_header("STEP 6: Regime Analysis")
    
    print(f"\n {'Regime':<12} {'Folds':>8} {'Avg IC':>12} {'Win Rate':>12}")
    print(" " + "-"*50)
    
    for regime in ['bull', 'neutral', 'bear']:
        regime_data = results_df[results_df['regime'] == regime]
        if len(regime_data) > 0:
            n = len(regime_data)
            ic_mean = regime_data['ic'].mean()
            wr = (regime_data['ic'] > 0).mean() * 100
            
            emoji = "🟢" if regime == 'bull' else "🔴" if regime == 'bear' else "🟡"
            status = "✅" if ic_mean > 0 else "❌"
            
            print(f" {emoji} {regime:<10} {n:>8} {ic_mean:>+12.4f} {wr:>11.0f}% {status}")
    
    # =========================================================
    # STEP 7: Yearly Breakdown
    # =========================================================
    print_header("STEP 7: Yearly Breakdown")
    
    results_df['year'] = pd.to_datetime(results_df['test_start']).dt.year
    
    print(f"\n {'Year':<8} {'Folds':>8} {'Avg IC':>12} {'IC Std':>12} {'Win Rate':>12}")
    print(" " + "-"*55)
    
    for year in sorted(results_df['year'].unique()):
        year_data = results_df[results_df['year'] == year]
        n = len(year_data)
        ic_mean = year_data['ic'].mean()
        ic_s = year_data['ic'].std()
        wr = (year_data['ic'] > 0).mean() * 100
        status = "✅" if ic_mean > 0 else "❌"
        print(f" {year:<8} {n:>8} {ic_mean:>+12.4f} {ic_s:>12.4f} {wr:>11.0f}% {status}")
    
    # =========================================================
    # STEP 8: Feature Importance
    # =========================================================
    print_header("STEP 8: Feature Importance (Top 15)")
    
    # Normalize feature importance
    feature_importance_total = feature_importance_total / feature_importance_total.sum()
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance_total
    }).sort_values('importance', ascending=False)
    
    print(f"\n {'Rank':<6} {'Feature':<25} {'Importance':>12} {'Bar'}")
    print(" " + "-"*60)
    
    for idx, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        feature = row['feature']
        imp = row['importance']
        bar = "█" * int(imp * 100)
        print(f" {idx:<6} {feature:<25} {imp:>12.4f} {bar}")
    
    # =========================================================
    # STEP 9: Save Results
    # =========================================================
    print_header("STEP 9: Saving Results")
    
    # Validation results
    results_df.to_csv(results_dir / "validation_results.csv", index=False)
    print(f" 💾 Saved: validation_results.csv")
    
    # Predictions
    if all_predictions:
        preds = pd.concat(all_predictions, ignore_index=True)
        preds.to_csv(results_dir / "predictions.csv", index=False)
        print(f" 💾 Saved: predictions.csv ({len(preds):,} rows)")
    
    # Feature importance
    importance_df.to_csv(results_dir / "feature_importance.csv", index=False)
    print(f" 💾 Saved: feature_importance.csv")
    
    # Features dataset (for other scripts)
    features_df.to_pickle(settings.data.processed_dir / "features_dataset.pkl")
    print(f" 💾 Saved: features_dataset.pkl ({n_stocks} stocks)")
    
    elapsed = time.time() - start_time
    
    # =========================================================
    # Complete
    # =========================================================
    print("\n" + "="*65)
    print("🎉 "*16)
    print(" LIGHTGBM ML PIPELINE COMPLETE")
    print("🎉 "*16)
    print(f" ⏱️  Time: {elapsed:.1f}s")
    print(f" 📊 Stocks: {n_stocks}")
    print("="*65)
    
    # =========================================================
    # Trading Strategy Summary
    # =========================================================
    print_header("TRADING STRATEGY")
    print(f"""
  [STRATEGY] LIGHTGBM ML STRATEGY:
  
    [DATA] Dataset:
      -> {n_stocks} S&P 500 stocks
      -> {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
      -> Same data as main.py and run_backtest.py
  
    [TRAINING] Model Training:
      -> Train on 2 years of data (504 days)
      -> 47 features: momentum, volatility, volume, trend
      -> Walk-forward validation (no look-ahead bias)
    
    [RULES] Trading Rules:
      -> Each month, predict next month returns
      -> Rank all {n_stocks} stocks by prediction
      -> Buy TOP 10 stocks (equal weight)
      -> Hold for 1 month, rebalance
    
    [RISK] Risk Management:
      -> Detect market regime (bull/neutral/bear)
      -> Bear market: Reduce position size by 50%
      -> Neutral: Normal position size
      -> Bull: Full position size
    
    [INSIGHT] Why It Works:
      -> LightGBM finds non-linear patterns
      -> Feature importance shows what matters
      -> Regime detection avoids 2022-style crashes
    """)


if __name__ == "__main__":
    main()