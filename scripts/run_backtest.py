#!/usr/bin/env python3
"""
Professional Quant Strategy Backtest
Regime-based long/short with multiple factors
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings, print_welcome


def print_header(title: str) -> None:
    """Print section header."""
    print(f"\n{'='*65}")
    print(f"  📊 {title}")
    print('='*65)


def detect_regime(market_df: pd.DataFrame, date) -> dict:
    """Simple but effective regime detection."""
    recent = market_df[market_df['date'] <= date].tail(200)
    
    if len(recent) < 50:
        return {'regime': 'neutral', 'confidence': 0.5}
    
    current = recent['close'].iloc[-1]
    ma_50 = recent['close'].tail(50).mean()
    ma_200 = recent['close'].mean()
    
    mom_20 = (current / recent['close'].iloc[-20] - 1) if len(recent) >= 20 else 0
    
    returns = recent['close'].pct_change().dropna()
    vol = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.15
    
    score = 0
    if current > ma_200: score += 2
    if current > ma_50: score += 1
    if mom_20 > 0.03: score += 1
    if mom_20 < -0.03: score -= 1
    if vol > 0.25: score -= 1
    if vol < 0.15: score += 1
    
    if score >= 3:
        regime = 'strong_bull'
    elif score >= 1:
        regime = 'bull'
    elif score <= -2:
        regime = 'bear'
    else:
        regime = 'neutral'
    
    return {'regime': regime, 'confidence': abs(score) / 5}


def main():
    print_welcome()
    
    print("\n" + "🔬 "*20)
    print("  PROFESSIONAL QUANT STRATEGY BACKTEST")
    print("🔬 "*20)
    
    settings = Settings(show_survivorship_warning=False)
    results_dir = settings.results_dir
    
    # =========================================================
    # Load Data
    # =========================================================
    print_header("STEP 1: Loading Data")
    
    features_path = settings.data.processed_dir / "features_dataset.pkl"
    features_df = pd.read_pickle(features_path)
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    panel_path = settings.data.panel_path
    prices = pd.read_pickle(panel_path)
    prices['date'] = pd.to_datetime(prices['date'])
    
    print(f"  📂 Features: {features_df.shape}")
    print(f"  📂 Prices: {prices.shape}")
    
    market = prices.groupby('date')['close'].mean().reset_index()
    
    # =========================================================
    # Create Factor Scores
    # =========================================================
    print_header("STEP 2: Creating Factor Scores")
    
    vol_cols = [c for c in features_df.columns if 'volatility' in c.lower() and 'rank' not in c]
    if vol_cols:
        features_df['vol_score'] = features_df[vol_cols].mean(axis=1)
    else:
        features_df['vol_score'] = 0.5
    
    mom_cols = [c for c in features_df.columns if c.startswith('mom_') and 'rank' not in c and 'accel' not in c]
    if mom_cols:
        features_df['mom_score'] = features_df[mom_cols].mean(axis=1)
    else:
        features_df['mom_score'] = 0
    
    if 'rsi_14' in features_df.columns:
        features_df['mr_score'] = 1 - features_df['rsi_14']
    elif 'rsi_21' in features_df.columns:
        features_df['mr_score'] = 1 - features_df['rsi_21']
    else:
        features_df['mr_score'] = 0.5
    
    features_df['quality_score'] = 1 - features_df['vol_score']
    
    print("  ✅ Created 4 factor scores:")
    print("    → vol_score (volatility)")
    print("    → mom_score (momentum)")
    print("    → mr_score (mean reversion)")
    print("    → quality_score (stability)")
    
    # =========================================================
    # Backtest Settings
    # =========================================================
    print_header("STEP 3: Configuration")
    
    initial_capital = 1_000_000
    top_n_long = 10
    top_n_short = 5
    cost_bps = 10
    
    print(f"  💵 Capital    : ${initial_capital:,}")
    print(f"  📈 Long Pos   : {top_n_long}")
    print(f"  📉 Short Pos  : {top_n_short}")
    print(f"  💰 Cost       : {cost_bps} bps")
    
    # =========================================================
    # Create Portfolios
    # =========================================================
    print_header("STEP 4: Creating Portfolios")
    
    features_df['year_month'] = features_df['date'].dt.to_period('M')
    rebalance_dates = features_df.groupby('year_month')['date'].min().unique()
    
    portfolios = []
    
    for rebal_date in sorted(rebalance_dates):
        regime_info = detect_regime(market, rebal_date)
        regime = regime_info['regime']
        
        day_df = features_df[features_df['date'] == rebal_date].copy()
        
        if len(day_df) < 15:
            continue
        
        if regime in ['strong_bull', 'bull']:
            day_df['long_score'] = day_df['vol_score'] * 0.6 + day_df['mom_score'] * 0.4
            do_short = (regime == 'strong_bull')
        elif regime == 'bear':
            day_df['long_score'] = day_df['quality_score'] * 0.6 + day_df['mr_score'] * 0.4
            do_short = True
        else:
            day_df['long_score'] = day_df['quality_score'] * 0.5 + day_df['mr_score'] * 0.5
            do_short = False
        
        long_stocks = day_df.nlargest(top_n_long, 'long_score')['ticker'].tolist()
        
        if do_short:
            if regime == 'bear':
                day_df['short_score'] = day_df['vol_score'] * 0.7 - day_df['mom_score'] * 0.3
            else:
                day_df['short_score'] = -day_df['mom_score']
            short_stocks = day_df.nlargest(top_n_short, 'short_score')['ticker'].tolist()
        else:
            short_stocks = []
        
        portfolios.append({
            'date': pd.Timestamp(rebal_date),
            'regime': regime,
            'long_stocks': long_stocks,
            'short_stocks': short_stocks
        })
    
    print(f"  ✅ Created {len(portfolios)} portfolios")
    print()
    print("  🎯 Regime Distribution:")
    regimes = [p['regime'] for p in portfolios]
    for r in ['strong_bull', 'bull', 'neutral', 'bear']:
        count = regimes.count(r)
        emoji = "🟢" if r == 'strong_bull' else "🟡" if r == 'bull' else "⚪" if r == 'neutral' else "🔴"
        print(f"    {emoji} {r:<12}: {count} months")
    
    # =========================================================
    # Calculate Returns
    # =========================================================
    print_header("STEP 5: Calculating Returns")
    
    prices_df = prices[['date', 'ticker', 'close']].copy()
    prices_df = prices_df.sort_values(['ticker', 'date'])
    prices_df['return'] = prices_df.groupby('ticker')['close'].pct_change()
    
    portfolio_value = initial_capital
    daily_values = []
    
    for i, port in enumerate(portfolios):
        rebal_date = port['date']
        long_stocks = port['long_stocks']
        short_stocks = port['short_stocks']
        regime = port['regime']
        
        if i + 1 < len(portfolios):
            next_rebal = portfolios[i + 1]['date']
        else:
            next_rebal = prices_df['date'].max()
        
        n_trades = len(long_stocks) + len(short_stocks)
        cost = portfolio_value * n_trades * 0.1 * (cost_bps / 10000)
        portfolio_value -= cost
        
        n_long = len(long_stocks)
        n_short = len(short_stocks)
        
        if regime == 'bear':
            long_exposure = 0.5
            short_exposure = 0.3
        elif regime == 'neutral':
            long_exposure = 0.7
            short_exposure = 0.0
        else:
            long_exposure = 0.7
            short_exposure = 0.2 if n_short > 0 else 0.0
        
        period_data = prices_df[
            (prices_df['date'] > rebal_date) & 
            (prices_df['date'] <= next_rebal)
        ]
        
        for date in sorted(period_data['date'].unique()):
            day_data = period_data[period_data['date'] == date]
            
            daily_ret = 0
            
            if n_long > 0:
                long_data = day_data[day_data['ticker'].isin(long_stocks)]
                if len(long_data) > 0:
                    long_ret = long_data['return'].mean()
                    if pd.notna(long_ret):
                        daily_ret += long_exposure * long_ret
            
            if n_short > 0:
                short_data = day_data[day_data['ticker'].isin(short_stocks)]
                if len(short_data) > 0:
                    short_ret = short_data['return'].mean()
                    if pd.notna(short_ret):
                        daily_ret -= short_exposure * short_ret
            
            if pd.notna(daily_ret):
                portfolio_value *= (1 + daily_ret)
            
            daily_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'daily_return': daily_ret if pd.notna(daily_ret) else 0,
                'regime': regime
            })
    
    results = pd.DataFrame(daily_values)
    results = results.sort_values('date')
    results = results[results['portfolio_value'].notna()]
    results = results[results['portfolio_value'] > 0]
    
    print(f"  ✅ Calculated {len(results)} daily returns")
    
    # =========================================================
    # Performance Metrics
    # =========================================================
    print_header("STEP 6: Performance Metrics")
    
    if len(results) == 0:
        print("  ❌ [ERROR] No valid results!")
        return
    
    final_value = results['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    
    days = (results['date'].max() - results['date'].min()).days
    years = days / 365.25
    annual_return = ((final_value / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
    
    daily_vol = results['daily_return'].std()
    annual_vol = daily_vol * np.sqrt(252) * 100
    
    rf = 0.04
    sharpe = (annual_return/100 - rf) / (annual_vol/100) if annual_vol > 0 else 0
    
    results['peak'] = results['portfolio_value'].cummax()
    results['dd'] = (results['portfolio_value'] - results['peak']) / results['peak']
    max_dd = results['dd'].min() * 100
    
    monthly = results.groupby(results['date'].dt.to_period('M'))['daily_return'].sum()
    win_rate = (monthly > 0).mean() * 100
    
    print(f"""
  ╔════════════════════════════════════════════════════════════╗
  ║  💹 BACKTEST RESULTS                                       ║
  ╠════════════════════════════════════════════════════════════╣
  ║  📅 Period             : {str(results['date'].min().date())} → {str(results['date'].max().date())}  ║
  ║  📊 Trading Days       : {len(results):>10,}                       ║
  ╠════════════════════════════════════════════════════════════╣
  ║  💵 Initial Capital    : ${initial_capital:>12,}                ║
  ║  💰 Final Value        : ${final_value:>12,.0f}                ║
  ║  📈 Total Return       : {total_return:>+12.1f}%                ║
  ║  📈 Annual Return      : {annual_return:>+12.1f}%                ║
  ╠════════════════════════════════════════════════════════════╣
  ║  📉 Annual Volatility  : {annual_vol:>12.1f}%                ║
  ║  🎯 Sharpe Ratio       : {sharpe:>12.2f}                   ║
  ║  ⚠️  Max Drawdown       : {max_dd:>12.1f}%                ║
  ║  🏆 Monthly Win Rate   : {win_rate:>12.1f}%                ║
  ╚════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================
    # Yearly Breakdown
    # =========================================================
    print_header("STEP 7: Yearly Performance")
    
    results['year'] = results['date'].dt.year
    
    print(f"\n  {'Year':<8} {'Return':>12} {'Sharpe':>10} {'Max DD':>10}")
    print("  " + "-"*45)
    
    for year in sorted(results['year'].unique()):
        yr = results[results['year'] == year].copy()
        
        if len(yr) < 2:
            continue
        
        yr_ret = (yr['portfolio_value'].iloc[-1] / yr['portfolio_value'].iloc[0] - 1) * 100
        yr_vol = yr['daily_return'].std() * np.sqrt(252) * 100
        yr_sharpe = (yr_ret/100 - rf) / (yr_vol/100) if yr_vol > 0 else 0
        
        yr['yr_peak'] = yr['portfolio_value'].cummax()
        yr['yr_dd'] = (yr['portfolio_value'] - yr['yr_peak']) / yr['yr_peak']
        yr_max_dd = yr['yr_dd'].min() * 100
        
        status = "✅" if yr_ret > 0 else "❌"
        print(f"  {year:<8} {yr_ret:>+11.1f}% {yr_sharpe:>10.2f} {yr_max_dd:>9.1f}% {status}")
    
    # =========================================================
    # Save Results
    # =========================================================
    print_header("STEP 8: Saving Results")
    
    results.to_csv(results_dir / "backtest_results.csv", index=False)
    print(f"  💾 Saved: backtest_results.csv")
    
    print("\n" + "="*65)
    print("🎉 "*16)
    print("  COMPLETE")
    print("🎉 "*16)
    print(f"  ✅ Backtest finished successfully")
    print("="*65)


if __name__ == "__main__":
    main()