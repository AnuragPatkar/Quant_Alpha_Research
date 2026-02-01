"""
ML-Based Multi-Factor Alpha Model
==================================
Main entry point for the complete pipeline.

Uses REAL S&P 500 data with:
- Regime Detection
- Long/Short Strategy
- Market Impact Model

Usage:
    python main.py                    # Full pipeline
    python main.py --quick            # Quick test with less data
    python main.py --skip-backtest    # Skip backtesting
    python main.py --help             # Show help
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import argparse
from pathlib import Path
from datetime import datetime
import traceback
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Imports
from config.settings import settings
from quant_alpha.data import DataLoader
from quant_alpha.features import compute_all_features
from quant_alpha.research import WalkForwardValidator
from quant_alpha.backtest import Backtester
from quant_alpha.backtest.market_impact import MarketImpactModel


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ML-Based Multi-Factor Alpha Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Full pipeline
    python main.py --quick            # Quick test mode
    python main.py --skip-backtest    # Skip backtesting step
    python main.py --no-plots         # Skip plot generation
    python main.py --long-only        # Long only strategy (no shorts)
        """
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick test mode (fewer stocks, shorter period)'
    )
    
    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='Skip backtesting step'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    
    parser.add_argument(
        '--save-features',
        action='store_true',
        help='Save computed features to CSV'
    )
    
    parser.add_argument(
        '--long-only',
        action='store_true',
        help='Use long-only strategy (no short positions)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


# =============================================================================
# BANNER & LOGGING
# =============================================================================

def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ML-BASED MULTI-FACTOR ALPHA MODEL                        ║
║     Cross-Sectional Stock Return Prediction                  ║
║                                                              ║
║     Data: Real S&P 500 (Yahoo/Stooq)                         ║
║     Model: LightGBM                                          ║
║     Strategy: Regime-Based Long/Short                        ║
║     Costs: Almgren-Chriss Market Impact                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_step(step_num: int, title: str):
    """Print step header."""
    print(f"\n{'='*60}")
    print(f"📌 STEP {step_num}: {title}")
    print(f"{'='*60}")


def print_success(message: str):
    """Print success message."""
    print(f"   ✅ {message}")


def print_error(message: str):
    """Print error message."""
    print(f"   ❌ {message}")


def print_warning(message: str):
    """Print warning message."""
    print(f"   ⚠️ {message}")


def print_info(message: str):
    """Print info message."""
    print(f"   ℹ️ {message}")


# =============================================================================
# REGIME DETECTION (Same as run_backtest.py)
# =============================================================================

def detect_regime(market_df: pd.DataFrame, date) -> dict:
    """
    Detect market regime: strong_bull, bull, neutral, bear.
    Same logic as run_backtest.py for consistency.
    """
    recent = market_df[market_df['date'] <= date].tail(200)
    
    if len(recent) < 50:
        return {'regime': 'neutral', 'confidence': 0.5, 'volatility': 0.15}
    
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
        regime = 'strong_bull'
    elif score >= 1:
        regime = 'bull'
    elif score <= -2:
        regime = 'bear'
    else:
        regime = 'neutral'
    
    return {'regime': regime, 'confidence': abs(score) / 5, 'volatility': vol}


# =============================================================================
# ADVANCED BACKTEST (Same as run_backtest.py)
# =============================================================================

def run_advanced_backtest(
    features_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    args,
    initial_capital: float = 1_000_000,
    top_n_long: int = 10,
    top_n_short: int = 5
) -> dict:
    """
    Run advanced backtest with:
    - Regime detection
    - Long/Short strategy
    - Market impact model
    
    Same logic as run_backtest.py for consistent results.
    """
    
    print("\n" + "="*60)
    print("💼 ADVANCED BACKTESTING (Regime-Based Long/Short)")
    print("="*60)
    
    # Initialize market impact model
    impact_model = MarketImpactModel(
        temp_impact_coef=50,
        perm_impact_coef=20,
        bid_ask_spread_bps=1.5
    )
    
    # If long-only mode
    if args.long_only:
        top_n_short = 0
        print_info("Long-only mode: No short positions")
    
    print(f"   💵 Capital    : ${initial_capital:,}")
    print(f"   📈 Long Pos   : {top_n_long}")
    print(f"   📉 Short Pos  : {top_n_short}")
    print(f"   💰 Cost Model : Almgren-Chriss Market Impact")
    
    # Create market index
    market = prices_df.groupby('date')['close'].mean().reset_index()
    
    # Calculate stock metrics for impact model
    stock_metrics = prices_df.groupby('ticker').agg({
        'close': 'last',
        'volume': 'mean',
    }).reset_index()
    stock_metrics.columns = ['ticker', 'last_price', 'avg_volume']
    stock_metrics['avg_dollar_volume'] = stock_metrics['last_price'] * stock_metrics['avg_volume']
    
    # Daily volatility per stock
    vol_df = prices_df.groupby('ticker').apply(
        lambda x: x['close'].pct_change().std() if len(x) > 1 else 0.02
    ).reset_index()
    vol_df.columns = ['ticker', 'daily_volatility']
    stock_metrics = stock_metrics.merge(vol_df, on='ticker', how='left')
    stock_metrics['daily_volatility'] = stock_metrics['daily_volatility'].fillna(0.02)
    
    print(f"   📊 Avg Daily Volume: ${stock_metrics['avg_dollar_volume'].mean():,.0f}")
    print(f"   📊 Avg Daily Vol: {stock_metrics['daily_volatility'].mean()*100:.2f}%")
    
    # Create factor scores (same as run_backtest.py)
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
    
    print_success("Created 4 factor scores: vol, mom, mr, quality")
    
    # Create portfolios
    features_df['year_month'] = features_df['date'].dt.to_period('M')
    rebalance_dates = features_df.groupby('year_month')['date'].min().unique()
    
    portfolios = []
    regime_counts = {'strong_bull': 0, 'bull': 0, 'neutral': 0, 'bear': 0}
    
    for rebal_date in sorted(rebalance_dates):
        regime_info = detect_regime(market, rebal_date)
        regime = regime_info['regime']
        regime_counts[regime] += 1
        
        day_df = features_df[features_df['date'] == rebal_date].copy()
        
        if len(day_df) < 15:
            continue
        
        # Regime-based scoring
        if regime in ['strong_bull', 'bull']:
            day_df['long_score'] = day_df['vol_score'] * 0.6 + day_df['mom_score'] * 0.4
            do_short = (regime == 'strong_bull') and (top_n_short > 0)
        elif regime == 'bear':
            day_df['long_score'] = day_df['quality_score'] * 0.6 + day_df['mr_score'] * 0.4
            do_short = (top_n_short > 0)
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
            'short_stocks': short_stocks,
            'volatility': regime_info.get('volatility', 0.15)
        })
    
    print_success(f"Created {len(portfolios)} monthly portfolios")
    
    # Print regime distribution
    print("\n   🎯 Regime Distribution:")
    for r in ['strong_bull', 'bull', 'neutral', 'bear']:
        emoji = "🟢" if r == 'strong_bull' else "🟡" if r == 'bull' else "⚪" if r == 'neutral' else "🔴"
        print(f"      {emoji} {r:<12}: {regime_counts[r]} months")
    
    # Prepare price data for returns
    prices_ret = prices_df[['date', 'ticker', 'close', 'volume']].copy()
    prices_ret = prices_ret.sort_values(['ticker', 'date'])
    prices_ret['return'] = prices_ret.groupby('ticker')['close'].pct_change()
    
    # Run backtest
    portfolio_value = initial_capital
    daily_values = []
    total_impact_cost = 0
    
    print("\n   📈 Running backtest...")
    
    for i, port in enumerate(portfolios):
        rebal_date = port['date']
        long_stocks = port['long_stocks']
        short_stocks = port['short_stocks']
        regime = port['regime']
        
        if i + 1 < len(portfolios):
            next_rebal = portfolios[i + 1]['date']
        else:
            next_rebal = prices_ret['date'].max()
        
        # Calculate market impact for trades
        all_stocks = long_stocks + short_stocks
        n_positions = len(all_stocks)
        
        if n_positions > 0:
            position_size = portfolio_value / n_positions
            period_impact_cost = 0
            
            for ticker in all_stocks:
                stock_data = stock_metrics[stock_metrics['ticker'] == ticker]
                
                if len(stock_data) > 0:
                    daily_dollar_volume = stock_data['avg_dollar_volume'].values[0]
                    stock_volatility = stock_data['daily_volatility'].values[0]
                    stock_price = stock_data['last_price'].values[0]
                else:
                    daily_dollar_volume = 500_000_000
                    stock_volatility = 0.02
                    stock_price = 100
                
                total_slippage, _ = impact_model.calculate_slippage(
                    order_size_dollars=position_size,
                    daily_volume_dollars=daily_dollar_volume,
                    volatility=stock_volatility,
                    price=stock_price
                )
                
                trade_cost = position_size * total_slippage
                period_impact_cost += trade_cost
            
            portfolio_value -= period_impact_cost
            total_impact_cost += period_impact_cost
        
        # Position exposures based on regime
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
        
        # Calculate daily returns
        period_data = prices_ret[
            (prices_ret['date'] > rebal_date) & 
            (prices_ret['date'] <= next_rebal)
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
    
    # Create results DataFrame
    results = pd.DataFrame(daily_values)
    results = results.sort_values('date')
    results = results[results['portfolio_value'].notna()]
    results = results[results['portfolio_value'] > 0]
    
    # Calculate metrics
    final_value = results['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital - 1)
    
    days = (results['date'].max() - results['date'].min()).days
    years = days / 365.25
    annual_return = ((final_value / initial_capital) ** (1/years) - 1) if years > 0 else 0
    
    daily_vol = results['daily_return'].std()
    annual_vol = daily_vol * np.sqrt(252)
    
    rf = 0.04
    sharpe = (annual_return - rf) / annual_vol if annual_vol > 0 else 0
    
    results['peak'] = results['portfolio_value'].cummax()
    results['dd'] = (results['portfolio_value'] - results['peak']) / results['peak']
    max_dd = results['dd'].min()
    
    monthly = results.groupby(results['date'].dt.to_period('M'))['daily_return'].sum()
    win_rate = (monthly > 0).mean()
    
    impact_drag = (total_impact_cost / initial_capital) / years if years > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'total_impact_cost': total_impact_cost,
        'annual_cost_drag': impact_drag,
        'trading_days': len(results),
        'portfolios': len(portfolios)
    }
    
    # Print results
    print(f"""
  ╔════════════════════════════════════════════════════════════╗
  ║  💹 BACKTEST RESULTS (WITH MARKET IMPACT)                  ║
  ╠════════════════════════════════════════════════════════════╣
  ║  📅 Period             : {str(results['date'].min().date())} → {str(results['date'].max().date())}  ║
  ║  📊 Trading Days       : {len(results):>10,}                       ║
  ╠════════════════════════════════════════════════════════════╣
  ║  💵 Initial Capital    : ${initial_capital:>12,}                ║
  ║  💰 Final Value        : ${final_value:>12,.0f}                ║
  ║  📈 Total Return       : {total_return*100:>+12.1f}%                ║
  ║  📈 Annual Return      : {annual_return*100:>+12.1f}%                ║
  ╠════════════════════════════════════════════════════════════╣
  ║  📉 Annual Volatility  : {annual_vol*100:>12.1f}%                ║
  ║  🎯 Sharpe Ratio       : {sharpe:>12.2f}                   ║
  ║  ⚠️  Max Drawdown       : {max_dd*100:>12.1f}%                ║
  ║  🏆 Monthly Win Rate   : {win_rate*100:>12.1f}%                ║
  ╠════════════════════════════════════════════════════════════╣
  ║  💸 Total Impact Cost  : ${total_impact_cost:>12,.0f}                ║
  ║  📉 Annual Cost Drag   : {impact_drag*100:>12.2f}%                ║
  ╚════════════════════════════════════════════════════════════╝
    """)
    
    return {
        'results': results,
        'metrics': metrics,
        'portfolios': portfolios,
        'regime_counts': regime_counts
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(val_results: pd.DataFrame, backtest_data: dict, importance: pd.DataFrame, save: bool = True):
    """Create and save result plots."""
    print("\n" + "="*60)
    print("📊 CREATING PLOTS")
    print("="*60)
    
    try:
        settings.create_dirs()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ML Alpha Model - Performance Analysis', fontsize=14, fontweight='bold')
        
        # 1. IC by Fold
        ax = axes[0, 0]
        ic_col = None
        for col in ['test_ic', 'ic', 'rank_ic', 'test_rank_ic']:
            if col in val_results.columns:
                ic_col = col
                break
        
        if ic_col:
            ic_values = val_results[ic_col].values
            colors = ['green' if x > 0 else 'red' for x in ic_values]
            ax.bar(range(len(ic_values)), ic_values, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.axhline(y=np.mean(ic_values), color='blue', linestyle='--',
                       label=f"Mean: {np.mean(ic_values):.4f}")
            ax.set_xlabel('Fold')
            ax.set_ylabel('IC')
            ax.set_title('Information Coefficient by Fold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Portfolio Value Over Time
        ax = axes[0, 1]
        if backtest_data is not None and 'results' in backtest_data:
            results = backtest_data['results']
            ax.plot(results['date'], results['portfolio_value'], linewidth=2, color='green')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value ($)')
            ax.set_title('Portfolio Value Over Time')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
            ax.grid(True, alpha=0.3)
            
            # Add drawdown shading
            results['peak'] = results['portfolio_value'].cummax()
            ax.fill_between(results['date'], results['portfolio_value'], results['peak'], 
                           alpha=0.3, color='red', label='Drawdown')
            ax.legend()
        
        # 3. Feature Importance
        ax = axes[1, 0]
        if 'importance_pct' in importance.columns:
            imp_col = 'importance_pct'
        elif 'importance' in importance.columns:
            imp_col = 'importance'
            importance = importance.copy()
            importance['importance_pct'] = importance['importance'] * 100
            imp_col = 'importance_pct'
        else:
            imp_col = importance.columns[1]  # Assume second column is importance
        
        top_15 = importance.head(15).sort_values(imp_col)
        ax.barh(top_15['feature'], top_15[imp_col], color='steelblue')
        ax.set_xlabel('Importance (%)')
        ax.set_title('Top 15 Feature Importance')
        
        # 4. Monthly Returns Distribution
        ax = axes[1, 1]
        if backtest_data is not None and 'results' in backtest_data:
            results = backtest_data['results']
            monthly_returns = results.groupby(results['date'].dt.to_period('M'))['daily_return'].sum() * 100
            ax.hist(monthly_returns, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='black', linewidth=1)
            ax.axvline(x=monthly_returns.mean(), color='red', linestyle='--',
                       label=f"Mean: {monthly_returns.mean():.2f}%")
            ax.set_xlabel('Monthly Return (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Monthly Returns Distribution')
            ax.legend()
        
        plt.tight_layout()
        
        if save:
            plot_path = settings.plots_dir / 'results.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print_success(f"Saved: {plot_path}")
        
        plt.close()
        
    except Exception as e:
        print_error(f"Plot creation failed: {e}")
        traceback.print_exc()


# =============================================================================
# REPORTING
# =============================================================================

def print_final_report(val_results: pd.DataFrame, backtest_data: dict, importance: pd.DataFrame):
    """Print final report."""
    print("\n" + "="*60)
    print("📄 FINAL REPORT")
    print("="*60)
    
    # Find IC column
    ic_col = None
    for col in ['test_ic', 'ic', 'rank_ic', 'test_rank_ic']:
        if col in val_results.columns:
            ic_col = col
            break
    
    if ic_col:
        mean_ic = val_results[ic_col].mean()
        std_ic = val_results[ic_col].std()
    else:
        mean_ic = 0
        std_ic = 0
    
    ir = mean_ic / (std_ic + 1e-10)
    
    # Find hit rate column
    hit_col = None
    for col in ['test_hit_rate', 'hit_rate']:
        if col in val_results.columns:
            hit_col = col
            break
    
    hit_rate = val_results[hit_col].mean() if hit_col else 0
    
    print(f"""
┌──────────────────────────────────────────────────────────────┐
│                    VALIDATION RESULTS                         │
├──────────────────────────────────────────────────────────────┤
│   Folds:              {len(val_results):>10}                              │
│   Mean IC:            {mean_ic:>10.4f}                              │
│   Std IC:             {std_ic:>10.4f}                              │
│   Information Ratio:  {ir:>10.4f}                              │
│   Mean Hit Rate:      {hit_rate:>10.1%}                              │
└──────────────────────────────────────────────────────────────┘
    """)
    
    if backtest_data is not None and 'metrics' in backtest_data:
        m = backtest_data['metrics']
        print(f"""
┌──────────────────────────────────────────────────────────────┐
│                    BACKTEST RESULTS                          │
├──────────────────────────────────────────────────────────────┤
│   Total Return:       {m['total_return']:>10.1%}                              │
│   Annual Return:      {m['annual_return']:>10.1%}                              │
│   Sharpe Ratio:       {m['sharpe_ratio']:>10.2f}                              │
│   Max Drawdown:       {m['max_drawdown']:>10.1%}                              │
│   Win Rate:           {m['win_rate']:>10.1%}                              │
│   Total Impact Cost:  ${m['total_impact_cost']:>10,.0f}                        │
│   Annual Cost Drag:   {m['annual_cost_drag']:>10.2%}                              │
└──────────────────────────────────────────────────────────────┘
        """)
    
    # Top features - print only once!
    print("\n📊 TOP 10 FEATURES:")
    print("─" * 40)
    
    if 'importance_pct' in importance.columns:
        imp_col = 'importance_pct'
    elif 'importance' in importance.columns:
        imp_col = 'importance'
    else:
        imp_col = importance.columns[1]
    
    for idx, (_, row) in enumerate(importance.head(10).iterrows(), 1):
        feat_name = row['feature']
        feat_imp = row[imp_col]
        if imp_col == 'importance':
            feat_imp = feat_imp * 100
        print(f"   {idx:2d}. {feat_name:<25} {feat_imp:>6.2f}%")


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step_load_data(args) -> pd.DataFrame:
    """Step 1: Load data."""
    print_step(1, "LOAD DATA")
    
    loader = DataLoader()
    data = loader.load()
    
    # Quick mode: use fewer stocks
    if args.quick:
        tickers = data['ticker'].unique()[:20]
        data = data[data['ticker'].isin(tickers)]
        print_warning(f"Quick mode: Using only {len(tickers)} stocks")
    
    print_success(f"Loaded {len(data):,} rows, {data['ticker'].nunique()} stocks")
    print_info(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    return data


def step_feature_engineering(data: pd.DataFrame, args) -> tuple:
    """Step 2: Feature engineering."""
    print_step(2, "FEATURE ENGINEERING")
    
    features_df = compute_all_features(data)
    
    # Auto-detect feature names
    non_feature_cols = {'date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 
                        'forward_return', 'vol_score', 'mom_score', 'mr_score', 
                        'quality_score', 'year_month', 'long_score', 'short_score'}
    feature_names = [c for c in features_df.columns if c not in non_feature_cols]
    
    print_success(f"Created {len(feature_names)} features")
    print_success(f"Feature matrix: {len(features_df):,} rows")
    
    if hasattr(args, 'save_features') and args.save_features:
        features_path = settings.results_dir / 'features.csv'
        features_df.to_csv(features_path, index=False)
        print_info(f"Saved features to: {features_path}")
    
    return features_df, feature_names


def step_validation(features_df: pd.DataFrame, feature_names: list, args) -> tuple:
    """Step 3: Walk-forward validation."""
    print_step(3, "WALK-FORWARD VALIDATION")
    
    validator = WalkForwardValidator(feature_names=feature_names)
    _ = validator.train_and_validate(features_df)
    
    # Convert FoldResult objects to DataFrame
    fold_dicts = []
    for fr in validator.fold_results:
        d = fr.to_dict()
        if 'metrics' in d and isinstance(d['metrics'], dict):
            metrics = d.pop('metrics')
            d.update(metrics)
        fold_dicts.append(d)
    
    val_results = pd.DataFrame(fold_dicts)
    predictions = validator.predict_out_of_sample(features_df)
    model = validator.get_latest_model()
    
    # Calculate mean IC
    ic_col = None
    for col in ['test_ic', 'ic', 'rank_ic']:
        if col in val_results.columns:
            ic_col = col
            break
    
    mean_ic = val_results[ic_col].mean() if ic_col else 0
    
    print_success(f"Validation complete: {len(val_results)} folds")
    print_success(f"Mean IC: {mean_ic:.4f}")
    
    return val_results, predictions, model


def step_feature_importance(model) -> pd.DataFrame:
    """Step 4: Extract feature importance."""
    print_step(4, "FEATURE IMPORTANCE")
    
    importance = model.get_feature_importance()
    
    print("\n   Top 10 Features:")
    
    if 'importance_pct' in importance.columns:
        imp_col = 'importance_pct'
    elif 'importance' in importance.columns:
        imp_col = 'importance'
    else:
        imp_col = importance.columns[1]
    
    for idx, (_, row) in enumerate(importance.head(10).iterrows(), 1):
        feat_name = row['feature']
        feat_imp = row[imp_col]
        if imp_col == 'importance':
            feat_imp = feat_imp * 100
        print(f"      {idx:2d}. {feat_name:<25} {feat_imp:>6.2f}%")
    
    return importance


def step_save_model(model):
    """Step 4b: Save model."""
    settings.create_dirs()
    model_path = settings.models_dir / 'alpha_model.pkl'
    model.save(str(model_path))
    print_success(f"Model saved: {model_path}")


def step_save_results(backtest_data: dict, val_results: pd.DataFrame, importance: pd.DataFrame):
    """Save all results to files."""
    print_step(7, "SAVING RESULTS")
    
    settings.create_dirs()
    
    # Save backtest results
    if backtest_data and 'results' in backtest_data:
        results_path = settings.results_dir / 'backtest_results.csv'
        backtest_data['results'].to_csv(results_path, index=False)
        print_success(f"Saved: {results_path}")
    
    # Save metrics
    if backtest_data and 'metrics' in backtest_data:
        metrics_path = settings.results_dir / 'backtest_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(backtest_data['metrics'], f, indent=2, default=str)
        print_success(f"Saved: {metrics_path}")
    
    # Save validation results
    val_path = settings.results_dir / 'validation_results.csv'
    val_results.to_csv(val_path, index=False)
    print_success(f"Saved: {val_path}")
    
    # Save feature importance
    imp_path = settings.results_dir / 'feature_importance.csv'
    importance.to_csv(imp_path, index=False)
    print_success(f"Saved: {imp_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main pipeline - runs everything from data loading to backtesting."""
    
    args = parse_args()
    print_banner()
    
    start_time = datetime.now()
    settings.print_config()
    
    if args.quick:
        print("\n⚡ QUICK MODE ENABLED - Using reduced dataset")
    
    try:
        # STEP 1: Load Data
        data = step_load_data(args)
        
        # STEP 2: Feature Engineering
        features_df, feature_names = step_feature_engineering(data, args)
        
        # STEP 3: Walk-Forward Validation
        val_results, predictions, model = step_validation(features_df, feature_names, args)
        
        # STEP 4: Feature Importance & Save Model
        importance = step_feature_importance(model)
        step_save_model(model)
        
        # STEP 5: Advanced Backtesting (Same as run_backtest.py)
        if args.skip_backtest:
            print_step(5, "BACKTESTING")
            print_warning("Backtesting skipped (--skip-backtest flag)")
            backtest_data = None
        else:
            backtest_data = run_advanced_backtest(
                features_df=features_df.copy(),
                prices_df=data.copy(),
                args=args,
                initial_capital=1_000_000,
                top_n_long=10,
                top_n_short=5
            )
        
        # STEP 6: Visualization
        print_step(6, "VISUALIZATION")
        if args.no_plots:
            print_warning("Plots skipped (--no-plots flag)")
        else:
            create_plots(val_results, backtest_data, importance)
        
        # STEP 7: Save Results
        step_save_results(backtest_data, val_results, importance)
        
        # STEP 8: Final Report
        print_final_report(val_results, backtest_data, importance)
        
        # Completion
        elapsed = datetime.now() - start_time
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"   ⏱️  Total Time: {elapsed}")
        print(f"\n   📁 Output Files:")
        print(f"      • {settings.models_dir / 'alpha_model.pkl'}")
        print(f"      • {settings.results_dir / 'backtest_results.csv'}")
        print(f"      • {settings.results_dir / 'backtest_metrics.json'}")
        print(f"      • {settings.results_dir / 'validation_results.csv'}")
        print(f"      • {settings.results_dir / 'feature_importance.csv'}")
        if not args.no_plots:
            print(f"      • {settings.plots_dir / 'results.png'}")
        
        return {
            'validation': val_results,
            'predictions': predictions,
            'backtest': backtest_data,
            'importance': importance,
            'model': model,
            'success': True
        }
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user (Ctrl+C)")
        return {'success': False, 'error': 'Interrupted'}
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ PIPELINE FAILED!")
        print("="*60)
        print(f"\n   Error: {e}")
        print("\n   Traceback:")
        traceback.print_exc()
        
        return {'success': False, 'error': str(e)}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = main()
    
    if results.get('success', False):
        sys.exit(0)
    else:
        sys.exit(1)