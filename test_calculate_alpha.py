"""
ALPHA METRICS — FINAL CORRECT VERSION
Reconstructs equity curve from sell-side PnL in detailed_trade_report
"""

import pandas as pd
import numpy as np
from scipy import stats
import os, logging, warnings
warnings.filterwarnings("ignore")

from quant_alpha.utils import setup_logging
setup_logging()
logger = logging.getLogger(__name__)
from config.settings import config

BENCHMARK_CACHE  = config.CACHE_DIR  / "benchmark_sp500.parquet"
TRADE_REPORT     = config.RESULTS_DIR / "trade_report_mean_variance.csv"
EQUITY_CSV_PATH  = config.RESULTS_DIR / "plots" / "mean_variance" / "equity_curve.csv"

INITIAL_CAPITAL  = 1_000_000
RF_ANNUAL        = 0.04
RF_DAILY         = RF_ANNUAL / 252

# ── Benchmark ─────────────────────────────────────────────────────────────────
def get_benchmark(start_date=None):
    if start_date is None:
        start_date = getattr(config, 'BACKTEST_START_DATE', "2018-01-01")

    if os.path.exists(BENCHMARK_CACHE):
        df  = pd.read_parquet(BENCHMARK_CACHE)
        df.index = pd.to_datetime(df.index)
        col = "Close" if "Close" in df.columns else df.columns[0]
        prices = df[col]
    else:
        import yfinance as yf
        logger.info(f"Downloading Benchmark from {start_date}...")
        df = yf.download("^GSPC", start=start_date, progress=False, auto_adjust=True)
        
        # FIX: Handle yfinance MultiIndex (e.g. ('Close', '^GSPC'))
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if 'Close' in df.columns.get_level_values(0):
                    df = df.xs('Close', level=0, axis=1)
            except Exception:
                pass
        
        # Ensure we have a 'Close' column
        if 'Close' not in df.columns and df.shape[1] == 1:
            df.columns = ['Close']
            
        df.index = pd.to_datetime(df.index)
        df[['Close']].to_parquet(BENCHMARK_CACHE)
        prices = df["Close"]
    ret = prices.pct_change().dropna()
    ret.name = "benchmark"
    logger.info(f"Benchmark loaded: {ret.index.min().date()} → {ret.index.max().date()}")
    return ret

# ── Equity Curve ──────────────────────────────────────────────────────────────
def load_equity_curve():
    # Option 1: CSV saved by updated pipeline
    if os.path.exists(EQUITY_CSV_PATH):
        logger.info(f"Loading equity curve CSV: {EQUITY_CSV_PATH}")
        ec  = pd.read_csv(EQUITY_CSV_PATH)
        ec['date'] = pd.to_datetime(ec['date'])
        ec  = ec.set_index('date').sort_index()
        col = next((c for c in ['total_value','portfolio_value','equity','nav']
                    if c in ec.columns), ec.select_dtypes('number').columns[0])
        logger.info(f"  Column: '{col}' | {ec[col].iloc[0]:,.0f} → {ec[col].iloc[-1]:,.0f}")
        return ec[col].pct_change().dropna().rename("strategy")

    # Option 2: Reconstruct from trade report
    if not os.path.exists(TRADE_REPORT):
        logger.error(f"Trade report not found: {TRADE_REPORT}")
        return None

    logger.info(f"Reconstructing equity from: {TRADE_REPORT}")
    df = pd.read_csv(TRADE_REPORT)
    df['date'] = pd.to_datetime(df['date'])

    # Only sell trades have realized PnL
    sells = df[df['side'] == 'sell'].copy()
    logger.info(f"  Sell trades with PnL: {len(sells):,}")

    # Daily realized PnL
    daily_pnl = sells.groupby('date')['pnl'].sum()

    # Build NAV: start at INITIAL_CAPITAL, add daily PnL
    # Get full date range from all trades
    all_dates = pd.bdate_range(
        start=df['date'].min(),
        end=df['date'].max()
    )
    nav = pd.Series(index=all_dates, dtype=float)
    nav.iloc[0] = INITIAL_CAPITAL

    for i, date in enumerate(all_dates[1:], 1):
        pnl_today = daily_pnl.get(date, 0.0)
        nav.iloc[i] = nav.iloc[i-1] + pnl_today

    daily_ret = nav.pct_change().dropna()
    daily_ret.name = "strategy"

    logger.info(f"  NAV: {nav.iloc[0]:,.0f} → {nav.iloc[-1]:,.0f} "
                f"| Total Return: {(nav.iloc[-1]/nav.iloc[0]-1):.2%}")
    return daily_ret

# ── Metrics ───────────────────────────────────────────────────────────────────
def calculate_metrics():
    strategy = load_equity_curve()
    if strategy is None:
        return

    bench = get_benchmark()
    bench = bench.loc[:strategy.index.max()]

    aligned = pd.concat([strategy, bench], axis=1).dropna()
    logger.info(f"Aligned: {len(aligned)} days | "
                f"{aligned.index.min().date()} → {aligned.index.max().date()}")

    if len(aligned) < 50:
        logger.error("Too few aligned days.")
        return

    ex_s = aligned["strategy"]  - RF_DAILY
    ex_b = aligned["benchmark"] - RF_DAILY

    beta, alpha_d, r_val, p_val, _ = stats.linregress(ex_b, ex_s)

    strat_ann = aligned["strategy"].mean()  * 252
    bench_ann = aligned["benchmark"].mean() * 252
    jensens   = strat_ann - (RF_ANNUAL + beta * (bench_ann - RF_ANNUAL))
    
    # CAGR Calculation (Geometric Mean)
    days = (aligned.index.max() - aligned.index.min()).days
    total_ret = (1 + aligned['strategy']).prod() - 1
    cagr = (1 + total_ret) ** (365.25 / days) - 1

    active  = aligned["strategy"] - aligned["benchmark"]
    te      = active.std() * np.sqrt(252)
    IR      = active.mean() * 252 / te if te > 0 else 0

    sharpe  = ex_s.mean() / ex_s.std() * np.sqrt(252)
    dn_std  = ex_s[ex_s < 0].std()
    sortino = ex_s.mean() / dn_std * np.sqrt(252) if dn_std > 0 else 0
    treynor = (strat_ann - RF_ANNUAL) / beta if beta != 0 else 0

    up   = aligned[aligned["benchmark"] > 0]
    dn   = aligned[aligned["benchmark"] < 0]
    up_c = up["strategy"].mean()  / up["benchmark"].mean()  * 100 if len(up) > 0 else 0
    dn_c = dn["strategy"].mean()  / dn["benchmark"].mean()  * 100 if len(dn) > 0 else 0

    # Max drawdown from NAV
    nav_idx = (1 + aligned["strategy"]).cumprod()
    roll_max = nav_idx.cummax()
    dd = (nav_idx - roll_max) / roll_max
    max_dd = dd.min()

    # Rolling 3M alpha
    roll = []
    for i in range(63, len(aligned)):
        w = aligned.iloc[i-63:i]
        sl, ic_, _, _, _ = stats.linregress(
            w["benchmark"] - RF_DAILY, w["strategy"] - RF_DAILY)
        roll.append({'date': aligned.index[i],
                     'rolling_alpha': ic_ * 252, 'rolling_beta': sl})
    roll_df = pd.DataFrame(roll).set_index('date') if roll else pd.DataFrame()

    # ── Report ────────────────────────────────────────────────────────────────
    S = "═" * 62
    D = "─" * 62

    print(f"\n{S}")
    print(f"  ALPHA METRICS — QUANT ALPHA RESEARCH (vs S&P 500)")
    print(f"{S}")
    src = "equity_curve.csv" if os.path.exists(EQUITY_CSV_PATH) else "trade report (reconstructed NAV)"
    print(f"  Source:  {src}")
    print(f"  Period:  {aligned.index.min().date()} → {aligned.index.max().date()} ({len(aligned)} days)")
    print(f"{D}")
    print(f"  [ ALPHA ]")
    print(f"  Jensen's Alpha (ann.) : {jensens:>+8.2%}   target > 5%")
    print(f"  Alpha p-value         : {p_val:>8.4f}   {'✅ Significant' if p_val < 0.05 else '⚠️  Not significant'}")
    print(f"  Active Return (ann.)  : {strat_ann - bench_ann:>+8.2%}")
    print(f"  CAGR                  : {cagr:>+8.2%}")
    print(f"{D}")
    print(f"  [ RISK ]")
    print(f"  Beta                  : {beta:>8.4f}   {'✅' if beta < 0.5 else '⚠️ '} target < 0.5")
    print(f"  Max Drawdown          : {max_dd:>8.2%}")
    print(f"  R² vs Benchmark       : {r_val**2:>8.4f}")
    print(f"  Tracking Error (ann.) : {te:>8.2%}")
    print(f"{D}")
    print(f"  [ RATIOS ]")
    print(f"  Sharpe Ratio          : {sharpe:>8.4f}")
    print(f"  Sortino Ratio         : {sortino:>8.4f}")
    print(f"  Information Ratio     : {IR:>8.4f}   {'🏆 Excellent' if IR > 1.0 else '✅ Good' if IR > 0.5 else '⚠️  Marginal'}")
    print(f"  Treynor Ratio         : {treynor:>8.4f}")
    print(f"{D}")
    print(f"  [ BENCHMARK COMPARISON ]")
    print(f"  Strategy (ann.)       : {strat_ann:>+8.2%}")
    print(f"  S&P 500 (ann.)        : {bench_ann:>+8.2%}")
    print(f"  Excess Return (ann.)  : {strat_ann - bench_ann:>+8.2%}")
    print(f"  Up Capture            : {up_c:>7.1f}%   {'✅' if up_c > 100 else '⚠️ '}")
    print(f"  Down Capture          : {dn_c:>7.1f}%   {'✅' if dn_c < 100 else '⚠️ '}")
    if not roll_df.empty:
        print(f"{D}")
        print(f"  [ ROLLING ALPHA (3-month) ]")
        print(f"  Mean Rolling Alpha    : {roll_df['rolling_alpha'].mean():>+8.2%}")
        print(f"  % Periods Positive    : {(roll_df['rolling_alpha'] > 0).mean():>8.1%}")
        print(f"  Best 3M Alpha         : {roll_df['rolling_alpha'].max():>+8.2%}")
        print(f"  Worst 3M Alpha        : {roll_df['rolling_alpha'].min():>+8.2%}")
    print(f"{S}\n")

    # Sanity check
    if abs(jensens) > 0.25:
        logger.warning(
            f"⚠️  Alpha {jensens:.1%} looks high — run pipeline again with updated "
            f"run_trainer_and_ensemble.py to get equity_curve.csv for accurate results."
        )

    # Save
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    pd.Series({
        "jensens_alpha": jensens, "beta": beta,
        "r_squared": r_val**2, "p_value": p_val,
        "information_ratio": IR, "sharpe": sharpe,
        "sortino": sortino, "treynor": treynor,
        "tracking_error": te, "max_drawdown": max_dd,
        "active_return_ann": strat_ann - bench_ann,
        "up_capture": up_c, "down_capture": dn_c,
        "cagr": cagr
    }).to_csv(config.RESULTS_DIR / "alpha_metrics.csv", header=["value"])
    logger.info(f"Saved → {config.RESULTS_DIR / 'alpha_metrics.csv'}")

    if not roll_df.empty:
        roll_df.to_csv(config.RESULTS_DIR / "rolling_alpha.csv")

if __name__ == "__main__":
    calculate_metrics()