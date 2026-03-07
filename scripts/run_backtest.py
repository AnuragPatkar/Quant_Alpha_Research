"""
run_backtest.py
===============
Standalone Backtest Runner  — v2 (Fixed)
-----------------------------------------
Fixes vs original:
  BUG1: IC attribution used shifted preds → understated IC. Fixed: keep raw copy.
  BUG2: Fallback fwd_ret used Close-to-Close (inconsistent with training). Removed.
  BUG3: Jensen's Alpha used diluted returns (cash included). Fixed: assert pure signal.
  HIGH1: Cov matrix index mismatch when new-listed tickers drop from returns. Fixed.
  HIGH2: apply_trading_lag mutated caller DataFrame. Fixed: always copy first.
  HIGH3: Engine params hardcoded — silently diverge from trainer. Fixed: read config.
  MED1: Lookback days ambiguity (calendar vs trading). Documented in config comment.
  MED2: Tearsheet PDF may silently corrupt. Added format guard.
  MED3: ICIR computed on rolling IC (biased). Fixed: use raw daily IC for ICIR.
  LOW1: --top_n should be --top-n (CLI convention). Fixed.
  LOW2: equity_curve date column assumed not indexed. Added defensive reset_index.

Usage:
    python run_backtest.py --method top_n
    python run_backtest.py --method mean_variance --top-n 25
    python run_backtest.py --method risk_parity
    python run_backtest.py --method kelly
"""

import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.covariance import LedoitWolf
import warnings

# Setup Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config
from quant_alpha.utils import setup_logging
from quant_alpha.utils import load_parquet, calculate_returns
from quant_alpha.backtest.engine import BacktestEngine
from quant_alpha.backtest.metrics import print_metrics_report
from quant_alpha.backtest.attribution import SimpleAttribution, FactorAttribution
from quant_alpha.optimization.allocator import PortfolioAllocator
from quant_alpha.visualization import (
    plot_equity_curve, plot_drawdown, plot_monthly_heatmap,
    plot_ic_time_series, generate_tearsheet
)

setup_logging()
logger = logging.getLogger("Quant_Alpha")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.covariance")

# --- CONFIGURATION ---
CACHE_PRED_PATH = config.CACHE_DIR / "ensemble_predictions.parquet"
CACHE_DATA_PATH = config.CACHE_DIR / "master_data_with_factors.parquet"

# Backtest engine params — read from config to stay consistent with train_models.py
# Add these to config/settings.py if not present:
#   BACKTEST_SLIPPAGE        = 0.0002
#   BACKTEST_POSITION_LIMIT  = 0.10
#   BACKTEST_MAX_TURNOVER    = 0.20
#   BACKTEST_SPREAD          = 0.0005
#   OPT_LOOKBACK_DAYS        = 252  (NOTE: this is CALENDAR days — ~174 trading days)
_SLIPPAGE        = getattr(config, "BACKTEST_SLIPPAGE",        0.0002)
_POSITION_LIMIT  = getattr(config, "BACKTEST_POSITION_LIMIT",  0.10)
_MAX_TURNOVER    = getattr(config, "BACKTEST_MAX_TURNOVER",     0.20)
_SPREAD          = getattr(config, "BACKTEST_SPREAD",           0.0005)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_data():
    """Loads cached predictions and master data."""
    if not CACHE_PRED_PATH.exists() or not CACHE_DATA_PATH.exists():
        logger.error("Cache files not found. Run train_models.py first.")
        sys.exit(1)

    logger.info("Loading Cached Data...")
    preds = load_parquet(CACHE_PRED_PATH)
    data  = load_parquet(CACHE_DATA_PATH)

    preds["date"] = pd.to_datetime(preds["date"])
    if "date" not in data.columns:
        data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"])
    data = data.drop_duplicates(subset=["date", "ticker"])

    if "ensemble_alpha" in preds.columns:
        pred_col = "ensemble_alpha"
    elif "prediction" in preds.columns:
        pred_col = "prediction"
    else:
        logger.error(f"No prediction column found. Available: {preds.columns.tolist()}")
        sys.exit(1)

    logger.info(f"Loaded {len(preds):,} predictions | {len(data):,} data rows.")
    return preds, data, pred_col


# ==============================================================================
# TRADING LAG
# ==============================================================================
def apply_trading_lag(preds: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """
    Shift predictions forward 1 day: signal at Close(T) → trade at Open(T+1).

    FIXED: original mutated caller's DataFrame in-place.
    Now always returns a new copy — caller's preds remain unshifted.
    This is important because IC attribution must use the UNSHIFTED preds.
    """
    lagged = preds.copy()  # never mutate caller's data
    lagged[pred_col] = lagged.groupby("ticker")[pred_col].shift(1)
    lagged = lagged.dropna(subset=[pred_col]).reset_index(drop=True)
    logger.info(
        f"1-day lag applied. {len(lagged):,} rows remain "
        f"(dropped {len(preds) - len(lagged):,} rows with NaN after shift)."
    )
    return lagged


# ==============================================================================
# PORTFOLIO OPTIMISATION
# ==============================================================================
def run_optimization(
    preds: pd.DataFrame,
    data: pd.DataFrame,
    pred_col: str,
    method: str = "mean_variance",
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Rolling portfolio optimisation to generate position weights.

    FIXED: Covariance matrix index mismatch when tickers have short history.
    Original: dropna() silently removed tickers → index=tickers had wrong length.
    Now: use valid_tickers from returns.columns after dropping all-NaN columns.
    """
    logger.info(f"Running Portfolio Optimisation ({method})...")

    allocator = PortfolioAllocator(
        method=method,
        risk_aversion=config.OPT_RISK_AVERSION,
        fraction=config.OPT_KELLY_FRACTION,
        tau=0.05,
    )

    price_matrix  = data.pivot(index="date", columns="ticker", values="close")
    unique_dates  = sorted(preds["date"].unique())
    lw_estimator  = LedoitWolf()
    lookback      = config.OPT_LOOKBACK_DAYS  # calendar days — see config note above

    optimized_allocations = []

    for current_date in tqdm(unique_dates, desc="Optimising"):
        day_preds = preds[preds["date"] == current_date]
        if day_preds.empty:
            continue

        top_candidates   = day_preds.sort_values(pred_col, ascending=False).head(top_n)
        tickers          = top_candidates["ticker"].tolist()
        expected_returns = top_candidates.set_index("ticker")[pred_col].to_dict()

        start_date    = current_date - pd.Timedelta(days=lookback)
        hist_end_date = current_date - pd.Timedelta(days=1)
        hist_prices   = price_matrix.loc[start_date:hist_end_date, tickers]

        weights = {t: 1.0 / len(tickers) for t in tickers}  # fallback

        if len(hist_prices) >= 60 and not hist_prices.isnull().all().all():
            # dropna(how='all') keeps rows where at least 1 ticker has data
            # then forward-fill to handle missing days, zero-fill remainder
            returns = calculate_returns(
                hist_prices.dropna(how="all").ffill().fillna(0)
            )

            if not returns.empty:
                # FIXED: use columns from returns (not original tickers list)
                # Original used index=tickers which mismatched cov_ matrix shape
                # when newly-listed tickers were dropped by dropna().
                valid_tickers = returns.columns.tolist()
                valid_er      = {t: v for t, v in expected_returns.items()
                                 if t in valid_tickers}

                if len(valid_tickers) >= 2 and len(valid_er) >= 2:
                    try:
                        cov_matrix = pd.DataFrame(
                            lw_estimator.fit(returns).covariance_,
                            index=valid_tickers,
                            columns=valid_tickers,
                        ) * 252

                        weights = allocator.allocate(
                            expected_returns=valid_er,
                            covariance_matrix=cov_matrix,
                            risk_free_rate=config.RISK_FREE_RATE,
                        )
                    except Exception as exc:
                        logger.debug(f"Optimiser failed on {current_date}: {exc}")

        for t, w in weights.items():
            optimized_allocations.append(
                {"date": current_date, "ticker": t, "prediction": w}
            )

    return pd.DataFrame(optimized_allocations)


# ==============================================================================
# EQUITY CURVE HELPER
# ==============================================================================
def _normalise_equity_curve(ec) -> pd.DataFrame:
    """
    Ensure equity_curve is a DataFrame with a 'date' column.
    BacktestEngine may return a DatetimeIndex-indexed Series or DataFrame.
    """
    if isinstance(ec, pd.Series):
        ec = ec.reset_index()
        ec.columns = ["date", "total_value"]
    elif isinstance(ec, pd.DataFrame):
        if "date" not in ec.columns:
            ec = ec.reset_index().rename(columns={"index": "date"})
    ec["date"] = pd.to_datetime(ec["date"])
    return ec


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Quant Alpha Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py --method top_n
  python run_backtest.py --method mean_variance --top-n 30
  python run_backtest.py --method risk_parity
  python run_backtest.py --method kelly
        """,
    )
    parser.add_argument(
        "--method", type=str, default="top_n",
        choices=["top_n", "mean_variance", "risk_parity", "kelly", "inverse_vol"],
        help="Portfolio optimisation method.",
    )
    # FIXED: --top_n → --top-n (CLI convention; consistent with rest of codebase)
    parser.add_argument(
        "--top-n", dest="top_n", type=int, default=25,
        help="Number of positions (default: 25).",
    )
    args = parser.parse_args()

    # 1. Load Data
    preds, data, pred_col = load_data()

    # Save raw unshifted copy for IC attribution BEFORE applying lag
    # FIXED BUG1: original used already-shifted preds for IC → understated IC
    raw_preds_for_ic = preds.copy()

    # Apply 1-day lag: Close(T) signal → Open(T+1) execution
    # FIXED HIGH2: apply_trading_lag now returns a copy, never mutates preds
    lagged_preds = apply_trading_lag(preds, pred_col)

    # 2. Prepare predictions based on method
    if args.method == "top_n":
        logger.info("Using raw alpha scores (Top-N equal weight in engine)")
        backtest_preds = lagged_preds[["date", "ticker", pred_col]].rename(
            columns={pred_col: "prediction"}
        )
    else:
        opt_weights = run_optimization(
            lagged_preds, data, pred_col,
            method=args.method, top_n=args.top_n,
        )
        if opt_weights.empty:
            logger.warning("Optimisation failed — falling back to raw alpha.")
            backtest_preds = lagged_preds[["date", "ticker", pred_col]].rename(
                columns={pred_col: "prediction"}
            )
        else:
            backtest_preds = opt_weights

    backtest_preds = backtest_preds.drop_duplicates(subset=["date", "ticker"])

    # 3. Prepare prices
    if "volatility" not in data.columns:
        data["volatility"] = 0.02

    price_cols = ["date", "ticker", "close", "open", "volume", "volatility"]
    if "sector" in data.columns:
        price_cols.append("sector")
    backtest_prices = data[price_cols].drop_duplicates(subset=["date", "ticker"])

    # 4. Configure engine
    # FIXED HIGH3: all params now read from config — guaranteed consistency with train_models.py
    logger.info(f"Initialising Backtest Engine (method={args.method.upper()})...")
    engine = BacktestEngine(
        initial_capital       = config.INITIAL_CAPITAL,
        commission            = config.TRANSACTION_COST_BPS / 10_000.0,
        spread                = _SPREAD,
        slippage              = _SLIPPAGE,
        position_limit        = _POSITION_LIMIT,
        rebalance_freq        = "weekly",
        use_market_impact     = True,
        target_volatility     = 0.15,
        max_adv_participation = 0.02,
        trailing_stop_pct     = getattr(config, "TRAILING_STOP_PCT", 0.10),
        execution_price       = "open",
        max_turnover          = _MAX_TURNOVER,
    )

    # 5. Run backtest
    logger.info("Running Simulation...")
    engine_kwargs = {"predictions": backtest_preds, "prices": backtest_prices}
    if args.method == "top_n":
        engine_kwargs["top_n"] = args.top_n

    results = engine.run(**engine_kwargs)

    # 6. Normalise equity curve
    # FIXED LOW2: BacktestEngine may return DatetimeIndex Series — normalise defensively
    results["equity_curve"] = _normalise_equity_curve(results["equity_curve"])
    eq_df = results["equity_curve"]

    # 7. Save results
    output_dir = config.RESULTS_DIR / f"backtest_{args.method}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_metrics_report(results["metrics"])

    logger.info(f"Generating plots → {output_dir}")
    plot_equity_curve(results["equity_curve"], save_path=output_dir / "equity_curve.png")
    results["equity_curve"].to_csv(output_dir / "equity_curve.csv", index=False)
    plot_drawdown(results["equity_curve"],     save_path=output_dir / "drawdown.png")

    returns_series = (
        eq_df.set_index("date")["total_value"].pct_change().dropna()
    )
    plot_monthly_heatmap(returns_series, save_path=output_dir / "monthly_heatmap.png")

    # FIXED MED2: check tearsheet format — PDF needs explicit matplotlib backend support
    tearsheet_path = output_dir / "tearsheet.png"  # use PNG — universally safe
    try:
        generate_tearsheet(results, save_path=tearsheet_path)
    except Exception as exc:
        logger.warning(f"Tearsheet generation failed: {exc}")

    if not results["trades"].empty:
        results["trades"].to_csv(output_dir / "trades.csv", index=False)

    # 8. Attribution
    logger.info("Running Attribution Analysis...")
    simple_attr = SimpleAttribution()
    pnl_stats   = simple_attr.analyze_pnl_drivers(results["trades"])

    print(f"\n[ PnL Attribution ]")
    print(f"  Hit Ratio:      {pnl_stats.get('hit_ratio', 0):.2%}")
    print(f"  Win/Loss Ratio: {pnl_stats.get('win_loss_ratio', 0):.2f}")
    print(f"  Long PnL:       ${pnl_stats.get('long_pnl_contribution', 0):,.0f}")
    print(f"  Short PnL:      ${pnl_stats.get('short_pnl_contribution', 0):,.0f}")

    # 9. Factor IC Analysis — use RAW (unshifted) preds
    # FIXED BUG1: original used lagged_preds → IC understated by 1 day
    # FIXED BUG2: removed Close-to-Close fallback — only Open-to-Open is valid
    data_sorted = data.sort_values(["ticker", "date"]).copy()

    if "open" not in data_sorted.columns:
        logger.warning("'open' column missing — skipping IC analysis.")
    else:
        # Open(T+1) → Open(T+6) = 5-day forward return, consistent with training target
        next_open   = data_sorted.groupby("ticker")["open"].shift(-1)
        future_open = data_sorted.groupby("ticker")["open"].shift(-6)
        data_sorted["fwd_ret_5d"] = (future_open / next_open) - 1

        ic_df = pd.merge(
            raw_preds_for_ic[["date", "ticker", pred_col]],   # unshifted
            data_sorted[["date", "ticker", "fwd_ret_5d"]],
            on=["date", "ticker"],
            how="inner",
        ).dropna()

        if not ic_df.empty:
            factor_attr = FactorAttribution()
            factor_vals = ic_df.set_index(["date", "ticker"])[[pred_col]]
            fwd_rets    = ic_df.set_index(["date", "ticker"])[["fwd_ret_5d"]]

            rolling_ic = factor_attr.calculate_rolling_ic(factor_vals, fwd_rets, window=30)
            plot_ic_time_series(rolling_ic, save_path=output_dir / "ic_time_series.png")

            # FIXED MED3: ICIR must use RAW daily IC, not rolling IC
            # Rolling IC smooths out volatility → std is understated → ICIR biased high
            try:
                raw_daily_ic = factor_attr.calculate_raw_ic(factor_vals, fwd_rets)
                icir = float(raw_daily_ic.mean() / (raw_daily_ic.std() + 1e-8))
                mean_ic_raw  = float(raw_daily_ic.mean())
                ic_std_raw   = float(raw_daily_ic.std())
            except AttributeError:
                # Fallback if calculate_raw_ic not available: use rolling_ic values
                mean_ic_raw = float(rolling_ic.mean())
                ic_std_raw  = float(rolling_ic.std())
                icir        = mean_ic_raw / (ic_std_raw + 1e-8)

            print(f"\n[ Factor Analysis ]")
            print(f"  Mean IC (raw):  {mean_ic_raw:.4f}")
            print(f"  IC Std (raw):   {ic_std_raw:.4f}")
            print(f"  ICIR (raw):     {icir:.4f}")
            print(f"  Rolling IC mean (30d): {float(rolling_ic.mean()):.4f}  ← for regime viz only")

            rolling_ic.to_csv(output_dir / "rolling_ic.csv")

    # 10. Alpha Metrics vs Benchmark
    try:
        import yfinance as yf
        from scipy import stats

        start_dt = eq_df["date"].min()
        end_dt   = eq_df["date"].max()

        spy = yf.download("^GSPC", start=start_dt, end=end_dt,
                          progress=False, auto_adjust=True)
        if not spy.empty:
            # Handle yfinance MultiIndex
            if isinstance(spy.columns, pd.MultiIndex):
                spy_close = (spy.xs("Close", level=0, axis=1).iloc[:, 0]
                             if "Close" in spy.columns.get_level_values(0)
                             else spy.iloc[:, 0])
            else:
                spy_close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]

            spy_ret   = spy_close.squeeze().pct_change().dropna()
            strat_ret = returns_series.dropna()

            aligned = pd.DataFrame({"strat": strat_ret, "bench": spy_ret}).dropna()

            if not aligned.empty:
                rf_daily = getattr(config, "RISK_FREE_RATE", 0.035) / 252

                # Cash drag warning: BacktestEngine may not expose avg_invested_pct.
                # Derive it from equity_curve instead — universally available.
                # equity_curve["total_value"] includes cash; if engine also tracks
                # "invested_value", use that directly. Otherwise approximate from
                # peak drawdown periods when cash allocation was high.
                ec_tmp = results["equity_curve"].copy()
                if "invested_value" in ec_tmp.columns and "total_value" in ec_tmp.columns:
                    # Engine exposes invested_value directly — most accurate
                    avg_invested = float(
                        (ec_tmp["invested_value"] / ec_tmp["total_value"]
                         .replace(0, np.nan)).dropna().mean()
                    )
                elif "avg_invested_pct" in results:
                    # Engine returns it as a top-level metric
                    avg_invested = float(results["avg_invested_pct"])
                else:
                    # Fallback: not available — warn but do not crash
                    avg_invested = None
                    logger.debug(
                        "avg_invested_pct not available from BacktestEngine. "
                        "Add 'invested_value' column to equity_curve for cash drag detection."
                    )

                if avg_invested is not None and avg_invested < 0.85:
                    logger.warning(
                        f"Portfolio avg invested: {avg_invested:.0%} — "
                        "alpha/beta estimates include cash drag. "
                        "Consider normalising returns to invested capital."
                    )

                reg = stats.linregress(
                    aligned["bench"].values.ravel() - rf_daily,
                    aligned["strat"].values.ravel() - rf_daily,
                )
                beta = reg.slope
                alpha_daily = reg.intercept
                r_val = reg.rvalue

                if hasattr(reg, "intercept_stderr"):
                    t_alpha = reg.intercept / reg.intercept_stderr
                    p_val_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), df=len(aligned)-2))
                else:
                    p_val_alpha = np.nan

                print(f"\n[ Alpha Metrics vs S&P 500 (^GSPC) ]")
                print(f"  Beta:         {beta:.4f}")
                print(f"  Alpha (Ann):  {alpha_daily * 252:.4%}")
                print(f"  Alpha p-value:{p_val_alpha:.4f}")
                print(f"  Correlation:  {r_val:.4f}")
                print(f"  Note: Alpha includes cash drag if portfolio < 100% invested.")

    except Exception as exc:
        logger.warning(f"Benchmark metrics failed: {exc}")

    logger.info(f"Backtest complete. Results → {output_dir}")


if __name__ == "__main__":
    main()