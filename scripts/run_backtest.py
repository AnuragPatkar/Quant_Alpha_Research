"""
Standalone Backtest Simulation Engine
=====================================
Executes a full historical simulation of a quantitative strategy based on
generated alpha signals.

Purpose
-------
This module serves as the **Strategy Validation Layer**, taking pre-generated alpha
signals (from `generate_predictions.py`) and simulating their performance under
realistic market conditions. It orchestrates the `BacktestEngine` to produce a
comprehensive set of performance metrics, visualizations, and attribution analyses.

Role in Quantitative Workflow
-----------------------------
Executed as the final out-of-sample stress test prior to capital deployment. It
validates the mathematical integrity of the target portfolio construction logic
against the raw generated alpha scores, accounting for transaction costs, market
impact, and execution slippage.

Usage
-----
.. code-block:: bash

    # 1. Top-N Equal Weight (simplest baseline)
    python scripts/run_backtest.py --method top_n --top-n 50

    # 2. Mean-Variance Optimization
    python scripts/run_backtest.py --method mean_variance --top-n 25

    # 3. Risk Parity (Equal Risk Contribution)
    python scripts/run_backtest.py --method risk_parity

Tools & Frameworks
------------------
-   **Ledoit-Wolf Shrinkage**: Guarantees positive-definite covariance matrix inversion.
-   **BacktestEngine**: Event-driven execution modeling with Almgren-Chriss impact logic.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

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
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_heatmap,
    plot_ic_time_series,
    generate_tearsheet,
)

setup_logging()
logger = logging.getLogger("Quant_Alpha")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.covariance")

CACHE_PRED_PATH = config.CACHE_DIR / "ensemble_predictions.parquet"
CACHE_DATA_PATH = config.CACHE_DIR / "master_data_with_factors.parquet"

_SLIPPAGE       = getattr(config, "BACKTEST_SLIPPAGE", getattr(config, "SLIPPAGE_PCT", 0.0005))
_POSITION_LIMIT = getattr(config, "BACKTEST_POSITION_LIMIT", getattr(config, "MAX_POSITION_SIZE", 0.10))
_MAX_TURNOVER   = getattr(config, "BACKTEST_MAX_TURNOVER", 0.20)
_SPREAD         = getattr(config, "BACKTEST_SPREAD", 0.0005)
_TARGET_VOL     = getattr(config, "BACKTEST_TARGET_VOL", 0.15)
_EXECUTION_PRICE = getattr(config, "EXECUTION_PRICE", "open").lower()
_REBALANCE_FREQ  = getattr(config, "REBALANCE_FREQ", "weekly").lower()
if _REBALANCE_FREQ == 'w': _REBALANCE_FREQ = 'weekly'
elif _REBALANCE_FREQ == 'd': _REBALANCE_FREQ = 'daily'
elif _REBALANCE_FREQ == 'm': _REBALANCE_FREQ = 'monthly'

# Enforces integer boundary resolution for precise trading-day rolling windows
_OPT_LOOKBACK_TD = int(getattr(config, "OPT_LOOKBACK_DAYS", 252))


def load_data():
    """
    Hydrates cached prediction and master data artifacts for simulation.

    Args:
        None

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, str]: A tuple containing the predictions 
            dataframe, the master dataset, and the target prediction column name.

    Raises:
        SystemExit: If required cached artifacts are missing or invalid.
    """
    if not CACHE_PRED_PATH.exists() or not CACHE_DATA_PATH.exists():
        logger.error(
            "Cache files not found. "
            "Run train_models.py and generate_predictions.py first."
        )
        sys.exit(1)

    logger.info("Loading cached data...")
    preds = load_parquet(CACHE_PRED_PATH)
    data  = load_parquet(CACHE_DATA_PATH)

    preds["date"] = pd.to_datetime(preds["date"])
    if "date" not in data.columns:
        data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"])
    data = data.drop_duplicates(subset=["date", "ticker"])
    
    # Temporal Gate: Binds boundaries strictly to the configured backtest environment
    start_date = pd.to_datetime(config.BACKTEST_START_DATE)
    preds = preds[preds["date"] >= start_date].reset_index(drop=True)
    data  = data[data["date"] >= start_date].reset_index(drop=True)

    if "ensemble_alpha" in preds.columns:
        pred_col = "ensemble_alpha"
    elif "prediction" in preds.columns:
        pred_col = "prediction"
    else:
        logger.error(
            f"No prediction column found. Available: {preds.columns.tolist()}"
        )
        sys.exit(1)

    logger.info(f"Loaded {len(preds):,} predictions | {len(data):,} data rows.")
    return preds, data, pred_col


def apply_trading_lag(preds: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """
    Applies a strict 1-day execution lag to alpha signals to prevent look-ahead bias.

    A signal generated using market data up to the Close of day T is only
    actionable at the Open of day T+1.

    Args:
        preds (pd.DataFrame): The raw alpha predictions mapped by date and ticker.
        pred_col (str): The column containing the predictive signal.

    Returns:
        pd.DataFrame: A copy of the predictions with signals shifted forward 
            by one trading day. The original dataframe remains intact for exact
            Signal-to-Return temporal alignment during IC analysis.
    """
    lagged = preds.copy()
    lagged[pred_col] = lagged.groupby("ticker")[pred_col].shift(1)
    lagged = lagged.dropna(subset=[pred_col]).reset_index(drop=True)
    logger.info(
        f"1-day lag applied. {len(lagged):,} rows remain "
        f"(dropped {len(preds) - len(lagged):,} rows with NaN after shift)."
    )
    return lagged


def run_optimization(
    preds: pd.DataFrame,
    data: pd.DataFrame,
    pred_col: str,
    method: str = "mean_variance",
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Executes rolling-window convex portfolio optimization to derive target weights.

    Args:
        preds (pd.DataFrame): The lagged prediction dataset.
        data (pd.DataFrame): The master historical OHLCV dataset.
        pred_col (str): The column identifying the alpha signal.
        method (str, optional): The optimization objective. Defaults to "mean_variance".
        top_n (int, optional): Maximum cardinality of the portfolio. Defaults to 25.

    Returns:
        pd.DataFrame: A longitudinal ledger mapping optimal fractional weights 
            for each target asset across the simulation horizon.
    """
    logger.info(f"Running Portfolio Optimisation ({method})...")

    allocator = PortfolioAllocator(
        method=method,
        risk_aversion=config.OPT_RISK_AVERSION,
        fraction=config.OPT_KELLY_FRACTION,
        tau=0.05,
    )

    price_matrix = data.pivot(index="date", columns="ticker", values="close")
    price_matrix = price_matrix.sort_index()
    unique_dates = sorted(preds["date"].unique())
    lw_estimator = LedoitWolf()

    # Defines explicit boundaries for positional constraints and limits
    _min_weight = getattr(config, "OPT_MIN_WEIGHT", 0.0)
    _max_weight = getattr(config, "OPT_MAX_WEIGHT", _POSITION_LIMIT)

    optimized_allocations = []

    for current_date in tqdm(unique_dates, desc="Optimising"):
        day_preds = preds[preds["date"] == current_date]
        if day_preds.empty:
            continue

        top_candidates   = day_preds.sort_values(pred_col, ascending=False).head(top_n)
        tickers          = top_candidates["ticker"].tolist()
        expected_returns = top_candidates.set_index("ticker")[pred_col].to_dict()

        weights = {t: 1.0 / len(tickers) for t in tickers}

        # Utilizes integer iloc offset for the covariance lookback window rather than 
        # calendar deltas to guarantee exact trading-day chronological alignment.
        try:
            loc = price_matrix.index.searchsorted(current_date)
        except Exception:
            continue

        start_iloc  = max(0, loc - _OPT_LOOKBACK_TD)
        hist_prices = price_matrix.iloc[start_iloc:loc][tickers]

        if len(hist_prices) >= 60 and not hist_prices.isnull().all().all():
            clean_prices = hist_prices.dropna(how="all").ffill()
            returns      = calculate_returns(clean_prices)

            # Isolates and drops sparse historical series with unresolved NaNs to prevent 
            # zero-biased variance artifacts during Ledoit-Wolf shrinkage.
            returns = returns.dropna(how="all").dropna(axis=1, how="any")

            if not returns.empty and returns.shape[1] >= 2:
                valid_tickers = returns.columns.tolist()
                valid_er      = {
                    t: v for t, v in expected_returns.items()
                    if t in valid_tickers
                }

                if len(valid_tickers) >= 2 and len(valid_er) >= 2:
                    # Remaps strict ordinal rank bounds [0, 1] to standardized return ranges
                    max_alpha = getattr(config, "MAX_ALPHA_RET", 0.30)
                    valid_er = {
                        t: float((v - 0.5) * 2.0 * max_alpha) 
                        for t, v in valid_er.items()
                    }

                    try:
                        cov_matrix = pd.DataFrame(
                            lw_estimator.fit(returns).covariance_,
                            index=valid_tickers,
                            columns=valid_tickers,
                        ) * 252

                        weights = allocator.allocate(
                            expected_returns=valid_er,
                            covariance_matrix=cov_matrix,
                            market_caps={t: 1e9 for t in valid_tickers},
                            confidence_level=getattr(config, "BL_CONFIDENCE_LEVEL", 0.6),
                            risk_free_rate=config.RISK_FREE_RATE,
                            min_weight=_min_weight,
                            max_weight=_max_weight,
                        )
                        
                        # Systemic Volatility Targeting matching exact execution methodology
                        target_vol = getattr(config, "BACKTEST_TARGET_VOL", 0.15)
                        if target_vol > 0 and method not in {"top_n", "kelly"}:
                            w_vec = np.array([weights.get(t, 0.0) for t in valid_tickers])
                            port_var = float(w_vec.T @ cov_matrix.values @ w_vec)
                            port_vol = np.sqrt(port_var) if port_var > 0 else 0.0
                            
                            if port_vol > 1e-6:
                                scaler = min(target_vol / port_vol, 3.0)
                                if scaler >= 1.0 or method not in {"risk_parity", "inverse_vol"}:
                                    scaled = {t: w * scaler for t, w in weights.items()}
                                    gross_exp = sum(abs(w) for w in scaled.values())
                                    max_lev = getattr(config, "MAX_LEVERAGE", 1.0)
                                    if gross_exp > max_lev:
                                        scaled = {t: w / (gross_exp / max_lev) for t, w in scaled.items()}
                                    for t, w in list(scaled.items()):
                                        if w > _max_weight: scaled[t] = _max_weight
                                        elif w < -_max_weight: scaled[t] = -_max_weight
                                    weights = scaled
                    except Exception as exc:
                        logger.warning(f"Optimiser failed on {current_date}: {exc}")

        for t, w in weights.items():
            optimized_allocations.append(
                {"date": current_date, "ticker": t, "prediction": w}
            )

    return pd.DataFrame(optimized_allocations)


def _normalise_equity_curve(ec) -> pd.DataFrame:
    """
    Defensively coerces engine equity curve outputs into a standardized DataFrame.

    Args:
        ec (pd.Series | pd.DataFrame): The raw equity curve output from the BacktestEngine.

    Returns:
        pd.DataFrame: A normalized DataFrame with explicit 'date' and 'total_value' columns.
    """
    if isinstance(ec, pd.Series):
        ec = ec.reset_index()
        ec.columns = ["date", "total_value"]
    elif isinstance(ec, pd.DataFrame):
        if "date" not in ec.columns:
            ec = ec.reset_index().rename(columns={"index": "date"})
    ec["date"] = pd.to_datetime(ec["date"])
    return ec


def main():
    """
    Primary orchestrator for historical strategy simulation.

    Parses execution constraints, loads required artifacts, executes the 
    BacktestEngine, and generates attribution metrics and teardown visualizations.

    Args:
        None

    Returns:
        None
    """
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
        "--method",
        type=str,
        default="top_n",
        choices=["top_n", "mean_variance", "risk_parity", "kelly", "inverse_vol"],
        help="Portfolio optimisation method.",
    )
    parser.add_argument(
        "--top-n",
        dest="top_n",
        type=int,
        default=25,
        help="Number of positions (default: 25).",
    )
    args = parser.parse_args()

    preds, data, pred_col = load_data()

    # Preserves un-shifted copy for strict point-in-time IC attribution
    raw_preds_for_ic = preds.copy()

    lagged_preds = apply_trading_lag(preds, pred_col)

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
            logger.warning("Optimization produced no weights — falling back to raw alpha.")
            backtest_preds = lagged_preds[["date", "ticker", pred_col]].rename(
                columns={pred_col: "prediction"}
            )
        else:
            backtest_preds = opt_weights

    backtest_preds = backtest_preds.drop_duplicates(subset=["date", "ticker"])

    if "volatility" not in data.columns:
        data["volatility"] = 0.02

    price_cols = ["date", "ticker", "close", "open", "volume", "volatility"]
    if "sector" in data.columns:
        price_cols.append("sector")
    backtest_prices = data[price_cols].drop_duplicates(subset=["date", "ticker"])

    # Align the execution price matrix exactly to the available Out-of-Sample boundaries
    if not backtest_preds.empty:
        first_signal_date = backtest_preds["date"].min()
        backtest_prices = backtest_prices[backtest_prices["date"] >= first_signal_date]

    engine = BacktestEngine(
        initial_capital       = config.INITIAL_CAPITAL,
        commission            = config.TRANSACTION_COST_BPS / 10_000.0,
        spread                = _SPREAD,
        slippage              = _SLIPPAGE,
        position_limit        = _POSITION_LIMIT,
        rebalance_freq        = _REBALANCE_FREQ,
        use_market_impact     = True,
        target_volatility     = _TARGET_VOL,
        max_adv_participation = 0.02,
        trailing_stop_pct     = getattr(config, "TRAILING_STOP_PCT", 0.10),
        execution_price       = _EXECUTION_PRICE,
        max_turnover          = _MAX_TURNOVER,
    )

    logger.info(
        f"Initialising BacktestEngine (method={args.method.upper()})..."
    )
    engine_kwargs = {
        "predictions": backtest_preds, 
        "prices": backtest_prices,
        "is_weights": (args.method != "top_n")
    }
    if args.method == "top_n":
        engine_kwargs["top_n"] = args.top_n

    results = engine.run(**engine_kwargs)

    results["equity_curve"] = _normalise_equity_curve(results["equity_curve"])
    eq_df = results["equity_curve"]

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

    tearsheet_path = output_dir / "tearsheet.png"
    try:
        generate_tearsheet(results, save_path=tearsheet_path)
    except Exception as exc:
        logger.warning(f"Tearsheet generation failed: {exc}")

    if not results["trades"].empty:
        results["trades"].to_csv(output_dir / "trades.csv", index=False)

    logger.info("Running Attribution Analysis...")
    simple_attr = SimpleAttribution()
    pnl_stats   = simple_attr.analyze_pnl_drivers(results["trades"])

    print("\n[ PnL Attribution ]")
    print(f"  Hit Ratio:      {pnl_stats.get('hit_ratio', 0):.2%}")
    print(f"  Win/Loss Ratio: {pnl_stats.get('win_loss_ratio', 0):.2f}")
    print(f"  Long PnL:       ${pnl_stats.get('long_pnl_contribution', 0):,.0f}")
    print(f"  Short PnL:      ${pnl_stats.get('short_pnl_contribution', 0):,.0f}")

    data_sorted = data.sort_values(["ticker", "date"]).copy()

    if "open" not in data_sorted.columns:
        logger.warning("'open' column missing — skipping IC analysis.")
    else:
        # 5-day forward return: strictly matches walk-forward Open(T+1) to Open(T+6) target structure
        next_open   = data_sorted.groupby("ticker")["open"].shift(-1)
        future_open = data_sorted.groupby("ticker")["open"].shift(-6)
        data_sorted["fwd_ret_5d"] = (future_open / next_open) - 1

        ic_df = pd.merge(
            raw_preds_for_ic[["date", "ticker", pred_col]], 
            data_sorted[["date", "ticker", "fwd_ret_5d"]],
            on=["date", "ticker"],
            how="inner",
        ).dropna()

        if not ic_df.empty:
            factor_attr = FactorAttribution()
            factor_vals = ic_df.set_index(["date", "ticker"])[[pred_col]]
            fwd_rets    = ic_df.set_index(["date", "ticker"])[["fwd_ret_5d"]]

            rolling_ic = factor_attr.calculate_rolling_ic(
                factor_vals, fwd_rets, window=30
            )
            plot_ic_time_series(
                rolling_ic, save_path=output_dir / "ic_time_series.png"
            )

            # Extracts daily ICs rather than rolling smoothed metrics to prevent variance deflation
            try:
                raw_daily_ic = factor_attr.calculate_raw_ic(factor_vals, fwd_rets)
                mean_ic_raw  = float(raw_daily_ic.mean())
                ic_std_raw   = float(raw_daily_ic.std())
                icir         = mean_ic_raw / (ic_std_raw + 1e-8)
            except AttributeError:
                # Fallback if calculate_raw_ic not implemented
                mean_ic_raw = float(rolling_ic.mean())
                ic_std_raw  = float(rolling_ic.std())
                icir        = mean_ic_raw / (ic_std_raw + 1e-8)

            print("\n[ Factor Analysis ]")
            print(f"  Mean IC (raw):         {mean_ic_raw:.4f}")
            print(f"  IC Std (raw):          {ic_std_raw:.4f}")
            print(f"  ICIR (raw):            {icir:.4f}")
            print(
                f"  Rolling IC mean (30d): {float(rolling_ic.mean()):.4f}"
                "  ← regime visualisation only"
            )

            rolling_ic.to_csv(output_dir / "rolling_ic.csv")

    try:
        import yfinance as yf
        from scipy import stats

        start_dt = eq_df["date"].min()
        end_dt   = eq_df["date"].max()

        spy = yf.download(
            "^GSPC", start=start_dt, end=end_dt,
            progress=False, auto_adjust=True,
        )

        if not spy.empty:
            if isinstance(spy.columns, pd.MultiIndex):
                spy_close = (
                    spy.xs("Close", level=0, axis=1).iloc[:, 0]
                    if "Close" in spy.columns.get_level_values(0)
                    else spy.iloc[:, 0]
                )
            else:
                spy_close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]

            spy_ret   = spy_close.squeeze().pct_change().dropna()
            strat_ret = returns_series.dropna()

            # Coerces timezone naivety to guarantee accurate benchmark indexing alignment
            spy_ret.index = pd.to_datetime(spy_ret.index).tz_localize(None)

            aligned = pd.DataFrame({"strat": strat_ret, "bench": spy_ret}).dropna()

            if not aligned.empty:
                rf_daily = getattr(config, "RISK_FREE_RATE", 0.035) / 252

                ec_tmp = results["equity_curve"].copy()
                if (
                    "invested_value" in ec_tmp.columns
                    and "total_value" in ec_tmp.columns
                ):
                    avg_invested = float(
                        (ec_tmp["invested_value"]
                         / ec_tmp["total_value"].replace(0, np.nan))
                        .dropna()
                        .mean()
                    )
                elif "avg_invested_pct" in results:
                    avg_invested = float(results["avg_invested_pct"])
                else:
                    avg_invested = None

                if avg_invested is not None and avg_invested < 0.85:
                    logger.warning(
                        f"Portfolio avg invested: {avg_invested:.0%} — "
                        "alpha/beta estimates include cash drag. "
                        "Consider normalising returns to invested capital."
                    )

                reg         = stats.linregress(
                    aligned["bench"].values.ravel() - rf_daily,
                    aligned["strat"].values.ravel() - rf_daily,
                )
                beta        = reg.slope
                alpha_daily = reg.intercept
                r_val       = reg.rvalue

                if hasattr(reg, "intercept_stderr") and reg.intercept_stderr:
                    t_alpha     = reg.intercept / reg.intercept_stderr
                    p_val_alpha = 2 * (
                        1 - stats.t.cdf(abs(t_alpha), df=len(aligned) - 2)
                    )
                else:
                    p_val_alpha = np.nan

                print("\n[ Alpha Metrics vs S&P 500 (^GSPC) ]")
                print(f"  Beta:          {beta:.4f}")
                print(f"  Alpha (Ann):   {alpha_daily * 252:.4%}")
                print(f"  Alpha p-value: {p_val_alpha:.4f}")
                print(f"  Correlation:   {r_val:.4f}")
                if avg_invested is not None:
                    print(
                        f"  Note: Avg invested = {avg_invested:.0%}. "
                        "Alpha includes cash drag."
                    )

    except Exception as exc:
        logger.warning(f"Benchmark metrics failed: {exc}")

    logger.info(f"Backtest complete. Results saved → {output_dir}")


if __name__ == "__main__":
    main()