#!/usr/bin/env python3
"""
Research Pipeline - Volatility Only Strategy
Simple rule-based approach, no ML overfitting
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings, print_welcome


def print_header(title: str) -> None:
    """Print section header."""
    print(f"\n{'='*65}")
    print(f"  📊 {title}")
    print('='*65)


def main():
    print_welcome()
    
    print("\n" + "🚀 "*20)
    print("  RESEARCH PIPELINE - VOLATILITY ONLY STRATEGY")
    print("🚀 "*20)
    
    start_time = time.time()
    
    settings = Settings(show_survivorship_warning=False)
    results_dir = settings.results_dir
    
    # =========================================================
    # STEP 1: Load Features
    # =========================================================
    print_header("STEP 1: Loading Features")
    
    features_path = settings.data.processed_dir / "features_dataset.pkl"
    
    if not features_path.exists():
        print("  ❌ [ERROR] Run: python scripts/run_research.py first")
        return
    
    df = pd.read_pickle(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  📂 Loaded: {df.shape[0]:,} rows")
    
    # =========================================================
    # STEP 2: Select Volatility Features
    # =========================================================
    print_header("STEP 2: Volatility-Only Strategy")
    
    volatility_features = [
        'volatility_10', 'volatility_21', 'volatility_63',
        'gk_volatility_21', 'gk_volatility_63',
        'hl_range_5', 'hl_range_21'
    ]
    
    feature_cols = [f for f in volatility_features if f in df.columns]
    
    print(f"  🎯 Using {len(feature_cols)} volatility features:")
    for f in feature_cols:
        print(f"    → {f}")
    
    # =========================================================
    # STEP 3: Create Volatility Score
    # =========================================================
    print_header("STEP 3: Creating Volatility Score")
    
    df['vol_score'] = df[feature_cols].mean(axis=1)
    
    print("  ✅ vol_score = average of volatility features")
    print("  💡 Logic: High vol stocks → Higher expected returns")
    
    # =========================================================
    # STEP 4: Walk-Forward Validation
    # =========================================================
    print_header("STEP 4: Walk-Forward Validation")
    
    test_df = df.dropna(subset=['forward_return', 'vol_score']).copy()
    dates = np.sort(test_df['date'].unique())
    
    start_date = pd.to_datetime('2022-01-01')
    dates = dates[dates >= np.datetime64(start_date)]
    
    test_days = 63
    step_days = 21
    
    print(f"\n  📋 Setup:")
    print(f"    📅 Test Period : {str(dates[0])[:10]} → {str(dates[-1])[:10]}")
    print(f"    📊 Test Window : 3 months")
    print(f"    🔄 Step Size   : 1 month")
    
    results = []
    all_predictions = []
    
    fold = 0
    idx = 0
    
    print(f"\n  {'Fold':<6} {'Period':<25} {'Samples':>8} {'IC':>10} {'Rank IC':>10}")
    print("  " + "-"*65)
    
    while idx + test_days <= len(dates):
        test_dates = dates[idx:idx + test_days]
        test_data = test_df[test_df['date'].isin(test_dates)]
        
        if len(test_data) < 100:
            idx += step_days
            continue
        
        predictions = test_data['vol_score'].values
        y_test = test_data['forward_return'].values
        
        try:
            ic = np.corrcoef(predictions, y_test)[0, 1]
            rank_ic = spearmanr(predictions, y_test)[0]
            ic = 0.0 if np.isnan(ic) else ic
            rank_ic = 0.0 if np.isnan(rank_ic) else rank_ic
        except:
            ic, rank_ic = 0.0, 0.0
        
        period = f"{str(test_dates[0])[:10]} → {str(test_dates[-1])[:10]}"
        
        results.append({
            'fold': fold,
            'test_start': str(test_dates[0])[:10],
            'test_end': str(test_dates[-1])[:10],
            'samples': len(test_data),
            'ic': ic,
            'rank_ic': rank_ic
        })
        
        status = "✅" if ic > 0 else "❌"
        print(f"  {fold:<6} {period}  {len(test_data):>8} {ic:>+10.4f} {rank_ic:>+10.4f} {status}")
        
        pred_df = test_data[['date', 'ticker', 'forward_return', 'vol_score']].copy()
        pred_df = pred_df.rename(columns={'vol_score': 'prediction'})
        all_predictions.append(pred_df)
        
        fold += 1
        idx += step_days
    
    # =========================================================
    # STEP 5: Results Summary
    # =========================================================
    print_header("STEP 5: Results Summary")
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("  ❌ [ERROR] No valid folds!")
        return
    
    avg_ic = results_df['ic'].mean()
    avg_rank_ic = results_df['rank_ic'].mean()
    ic_std = results_df['ic'].std()
    ir = avg_ic / ic_std if ic_std > 0 else 0
    win_rate = (results_df['ic'] > 0).mean() * 100
    
    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║  📈 VOLATILITY STRATEGY RESULTS                          ║
  ╠══════════════════════════════════════════════════════════╣
  ║  📊 Total Folds        : {len(results_df):<10}                     ║
  ║  📅 Period             : {results_df['test_start'].min()} → {results_df['test_end'].max()} ║
  ╠══════════════════════════════════════════════════════════╣
  ║  🎯 Average IC         : {avg_ic:>+10.4f}                     ║
  ║  🎯 Average Rank IC    : {avg_rank_ic:>+10.4f}                     ║
  ║  📉 IC Std Dev         : {ic_std:>10.4f}                     ║
  ║  💹 Information Ratio  : {ir:>+10.4f}                     ║
  ║  🏆 Win Rate           : {win_rate:>10.1f}%                    ║
  ╚══════════════════════════════════════════════════════════╝
    """)
    
    print("  📋 Interpretation:")
    print("  " + "-"*50)
    
    if avg_ic > 0.03:
        print("  ✅ [GOOD] Volatility premium is working!")
    elif avg_ic > 0:
        print("  🟡 [WEAK] Small positive signal")
    else:
        print("  ❌ [POOR] Volatility factor not working")
    
    if win_rate >= 60:
        print(f"  ✅ [RELIABLE] {win_rate:.0f}% months positive")
    elif win_rate >= 50:
        print(f"  🟡 [OK] {win_rate:.0f}% months positive")
    else:
        print(f"  ❌ [UNRELIABLE] Only {win_rate:.0f}% months positive")
    
    # =========================================================
    # STEP 6: Yearly Breakdown
    # =========================================================
    print_header("STEP 6: Yearly Breakdown")
    
    results_df['year'] = pd.to_datetime(results_df['test_start']).dt.year
    
    print(f"\n  {'Year':<8} {'Folds':>8} {'Avg IC':>12} {'IC Std':>12} {'Win Rate':>12}")
    print("  " + "-"*55)
    
    for year in sorted(results_df['year'].unique()):
        year_data = results_df[results_df['year'] == year]
        n = len(year_data)
        ic_mean = year_data['ic'].mean()
        ic_s = year_data['ic'].std()
        wr = (year_data['ic'] > 0).mean() * 100
        status = "✅" if ic_mean > 0 else "❌"
        print(f"  {year:<8} {n:>8} {ic_mean:>+12.4f} {ic_s:>12.4f} {wr:>11.0f}% {status}")
    
    # =========================================================
    # STEP 7: Save Results
    # =========================================================
    print_header("STEP 7: Saving Results")
    
    results_df.to_csv(results_dir / "validation_results_volatility.csv", index=False)
    print(f"  💾 Saved: validation_results_volatility.csv")
    
    if all_predictions:
        preds = pd.concat(all_predictions, ignore_index=True)
        preds.to_csv(results_dir / "predictions_volatility.csv", index=False)
        print(f"  💾 Saved: predictions_volatility.csv ({len(preds):,} rows)")
    
    elapsed = time.time() - start_time
    
    # =========================================================
    # Complete
    # =========================================================
    print("\n" + "="*65)
    print("🎉 "*16)
    print("  COMPLETE")
    print("🎉 "*16)
    print(f"  ⏱️  Time: {elapsed:.1f}s")
    print("="*65)
    
    # =========================================================
    # Trading Strategy Notes
    # =========================================================
    print("\n" + "="*65)
    print("  💡 TRADING STRATEGY")
    print("="*65)
    print("""
  📈 VOLATILITY RISK PREMIUM:
  
    → Each month, rank all stocks by volatility score
    → Buy TOP 10 highest volatility stocks (equal weight)
    → Hold for 1 month, rebalance
   
  🎯 WHY IT WORKS:
  
    → High volatility = Higher risk = Higher expected return
    → This is compensation for bearing risk
    → Works best in trending/bull markets
   
  ⚠️  CAUTION:
  
    → May have large drawdowns in crashes
    → 2022 bear market showed poor performance
    → Consider combining with momentum filter
    """)


if __name__ == "__main__":
    main()