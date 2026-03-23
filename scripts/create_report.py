"""
Executive Quantitative Research Report Generator
================================================
Aggregates insights from the alpha research pipeline into a management-level summary
to enforce quantitative observability.

Purpose
-------
This script acts as the final reporting layer of the quantitative research platform.
It generates an immutable, holistic audit of the system state, evaluating:
1.  **Data Integrity**: Ingestion latency and completeness (Prices, Fundamentals, Macro).
2.  **Factor Efficacy**: Statistical strength (Information Coefficient, t-stat) of the alpha library.
3.  **Model Robustness**: Out-of-sample performance of ML models (Annualized ICIR, Tiering).
4.  **Strategy Performance**: Walk-forward backtest metrics (Sharpe, Sortino, MaxDD) vs benchmark.
5.  **Portfolio Positioning**: Current target allocations, leverage bounds, and concentration risk.

Usage:
-----
Intended for daily execution post-pipeline or ad-hoc review.

.. code-block:: bash

    # Generate report to console
    python scripts/create_report.py

    # Save report to Markdown for distribution
    python scripts/create_report.py --output-file results/executive_summary.md

Importance
----------
-   **Operational Risk Mitigation**: Proactively identifies stale data or signal decay
    before trades are generated.
-   **Performance Attribution**: Decomposes returns into Alpha and Beta components,
    ensuring transparent attribution.
-   **Model Governance**: Formalizes visibility on production versus ensemble tiers
    by exposing the statistical gates required for promotion.
"""

import sys
import random
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config
from quant_alpha.utils import setup_logging
setup_logging()

logger = logging.getLogger("Quant_Alpha")

# In-memory cache for benchmark indexing to bypass redundant network calls
_SPY_CACHE: dict = {}

def weighted_symmetric_mae(y_true, y_pred):
    """
    Computes a custom asymmetric loss gradient and hessian.

    Calculates an objective that strictly penalizes directional sign errors
    over absolute magnitude. Exposed globally to ensure deterministic 
    resolution during Joblib unpickling of LightGBM/XGBoost artifacts.

    .. math:: L(y, \\hat{y}) = -w \\cdot \\tanh(y - \\hat{y})

    Args:
        y_true (np.ndarray): The ground truth target values.
        y_pred (np.ndarray): The model's predicted values.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the first-order 
            derivative (gradient) and second-order derivative (hessian).
    """
    residuals = y_true - y_pred
    weights   = np.where(y_true * y_pred < 0, 2.0, 1.0)
    grad      = -weights * np.tanh(residuals)
    hess      = np.maximum(weights * (1.0 - np.tanh(residuals) ** 2), 1e-3)
    return grad, hess

# Namespace injection to guarantee custom objective deserialization integrity
import sys as _sys
_sys.modules["__main__"].weighted_symmetric_mae = weighted_symmetric_mae  # type: ignore

def _format_currency(x):
    """
    Formats a numeric value into a USD currency string.

    Args:
        x (float): The numeric value to format.

    Returns:
        str: Formatted currency string, or "-" if the input is NaN.
    """
    if pd.isna(x): return "-"
    return f"${x:,.0f}"

def _format_pct(x, prec=2):
    """
    Formats a numeric value as a percentage string with a specified precision.

    Args:
        x (float): The numeric value to format (e.g., 0.05 for 5%).
        prec (int, optional): The number of decimal places. Defaults to 2.

    Returns:
        str: Formatted percentage string with explicit sign, or "-" if the input is NaN.
    """
    if pd.isna(x): return "-"
    return f"{x:+.{prec}%}"

def _format_float(x, prec=2):
    """
    Formats a numeric value as a float string with a specified precision.

    Args:
        x (float): The numeric value to format.
        prec (int, optional): The number of decimal places. Defaults to 2.

    Returns:
        str: Formatted float string, or "-" if the input is NaN.
    """
    if pd.isna(x): return "-"
    return f"{x:.{prec}f}"

def _trading_days_old(signal_date) -> int | None:
    """
    Calculates the temporal business day lag between generation and execution.

    Args:
        signal_date (str | datetime): The timestamp to evaluate.

    Returns:
        int | None: The number of trading days elapsed, or None if parsing fails.
    """
    try:
        sig = pd.Timestamp(signal_date)
        today = pd.Timestamp(datetime.now().date())
        bdays = pd.bdate_range(sig, today)
        return max(0, len(bdays) - 1)
    except Exception:
        return None

def _calc_cagr(series: pd.Series) -> float:
    """
    Computes Compound Annual Growth Rate (CAGR).
    
    .. math:: CAGR = (1 + R_{total})^{\\frac{365}{days}} - 1
    
    Args:
        series (pd.Series): The daily equity curve or portfolio value time series.

    Returns:
        float: The annualized CAGR. Returns 0.0 for structurally invalid lengths.
    """
    if len(series) < 2: return 0.0
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    cal_days = (series.index[-1] - series.index[0]).days
    if cal_days <= 0: return 0.0
    return (1 + total_ret) ** (365.0 / cal_days) - 1

def _calc_max_dd(series: pd.Series) -> float:
    """
    Calculates Maximum Drawdown.
    
    .. math:: MaxDD = \\min_t \\left( \\frac{NAV_t}{HWM_t} - 1 \\right)
    
    Args:
        series (pd.Series): The cumulative portfolio value or equity curve.

    Returns:
        float: The maximum peak-to-trough contraction percentage.
    """
    if len(series) < 1: return 0.0
    peak = series.cummax()
    return ((series / peak) - 1).min()

def _calc_sharpe(series: pd.Series) -> float:
    """
    Calculates the Annualized Sharpe Ratio.
    
    .. math:: SR = \\frac{\\mu_{ret}}{\\sigma_{ret}} \\sqrt{252}

    Args:
        series (pd.Series): The daily equity curve or portfolio value time series.

    Returns:
        float: The annualized Sharpe ratio. Returns 0.0 if there is insufficient
            data or zero volatility.
    """
    if len(series) < 2: return 0.0
    rets = series.pct_change().dropna()
    # Stability guard: Prevents DivisionByZero exceptions during flat market regimes
    if rets.std() < 1e-9: return 0.0
    return (rets.mean() / rets.std()) * np.sqrt(252)

def _calc_sortino(series: pd.Series, rf_annual: float = 0.035) -> float:
    """
    Calculates the Annualized Sortino Ratio.

    Unlike the Sharpe Ratio, this metric penalizes only downside volatility 
    relative to a Minimum Acceptable Return (MAR).

    .. math:: Sortino = \\frac{E[R_p - R_f]}{\\sigma_{down}} \\sqrt{252}

    Args:
        series (pd.Series): The daily equity curve or portfolio value time series.
        rf_annual (float, optional): The annualized risk-free rate. Defaults to 0.035.

    Returns:
        float: The annualized Sortino ratio. Returns 0.0 if there is insufficient
            data or zero downside variance.
    """
    if len(series) < 2: return 0.0
    rets  = series.pct_change().dropna()
    rf_d  = rf_annual / 252
    excess = rets - rf_d
    
    # Asymmetric Risk Profile: Isolates the Root Mean Square of negative deviations
    downside_sq = np.minimum(0, excess) ** 2
    tdd = np.sqrt(downside_sq.mean())
    
    # Stability guard: Prevents DivisionByZero exceptions during micro-variance regimes
    if tdd < 1e-9: return 0.0
    return (excess.mean() / tdd) * np.sqrt(252)

def _get_top_drawdowns(series: pd.Series, n: int = 3) -> list:
    """
    Identifies distinct historical drawdown periods and isolates the most severe.

    Args:
        series (pd.Series): The cumulative portfolio value or equity curve.
        n (int, optional): The maximum number of distinct drawdown periods to return. 
            Defaults to 3.

    Returns:
        list: A sequence of dictionaries detailing the depth, start date, 
            valley date, recovery date, and duration of the worst drawdowns.
    """
    if len(series) < 2:
        return []

    peak    = series.cummax()
    dd_curve = (series / peak) - 1

    drawdowns = []
    in_dd     = False
    start_d   = None
    min_val   = 0.0
    valley_d  = None
    
    # Epsilon applied to recovery detection to prevent infinite drawdown states 
    # caused by floating-point precision mismatches at historical high-water marks.
    RECOVERY_THRESHOLD = -0.001

    for date, val in dd_curve.items():
        if val < RECOVERY_THRESHOLD:
            if not in_dd:
                in_dd    = True
                start_d  = date
                min_val  = val
                valley_d = date
            elif val < min_val:
                min_val  = val
                valley_d = date
        else:
            if in_dd:
                in_dd = False
                drawdowns.append({
                    "depth":    min_val,
                    "start":    start_d,
                    "valley":   valley_d,
                    "recovery": date,
                    "days":     (date - start_d).days,
                })

    # Still in drawdown at end of series
    if in_dd:
        drawdowns.append({
            "depth":    min_val,
            "start":    start_d,
            "valley":   valley_d,
            "recovery": pd.NaT,
            "days":     (series.index[-1] - start_d).days,
        })

    # Ascending sort evaluates the most negative (severe) magnitudes first
    drawdowns.sort(key=lambda x: x["depth"])
    return drawdowns[:n]

def _fetch_spy(start: str, end: str) -> pd.Series | None:
    """
    Ingests contiguous S&P 500 daily returns for relative performance benchmarking.

    Args:
        start (str): Boundary inception date mapping.
        end (str): Boundary termination date mapping.

    Returns:
        pd.Series | None: Alignable daily percentage returns, or None upon API timeout.
    """
    key = f"{start}_{end}"
    if key in _SPY_CACHE:
        return _SPY_CACHE[key]
    try:
        import yfinance as yf
        spy = yf.download("^GSPC", start=start, end=end,
                          progress=False, auto_adjust=True)
        if spy.empty:
            return None
            
        # Resolves dynamic indexing constraints inherited from modern upstream APIs
        if isinstance(spy.columns, pd.MultiIndex):
            close = spy.xs("Close", level=0, axis=1).iloc[:, 0]
        else:
            close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]
        ret = close.squeeze().pct_change().dropna()
        _SPY_CACHE[key] = ret
        return ret
    except Exception:
        return None

class QuantitativeManagerReport:
    def __init__(self):
        self.report_date = datetime.now()
        self.sections    = []

    def add_section(self, title: str, content: str):
        self.sections.append((title, content))

    def check_data_health(self):
        """
        Audits the Data Warehouse for staleness and structural integrity.
        
        Evaluates the freshness of OHLCV prices, fundamental statements, and
        the active generation latency of the production alpha signals.

        Args:
            None

        Returns:
            None
        """
        lines = []

        price_files = sorted(config.PRICES_DIR.glob("*.csv"))
        if not price_files:
            lines.append("❌ Price Data: No files found.")
        else:
            sample_files = random.sample(price_files, min(5, len(price_files)))
            last_dates = []
            for pf in sample_files:
                try:
                    tmp = pd.read_csv(pf, nrows=5)
                    if "date" in tmp.columns:
                        full = pd.read_csv(pf, usecols=["date"],
                                           parse_dates=["date"])
                        last_dates.append(full["date"].max().date())
                    elif tmp.index.dtype == "datetime64[ns]" or tmp.index.name == "date":
                        full = pd.read_csv(pf, index_col=0, parse_dates=True)
                        last_dates.append(full.index.max().date())
                except Exception:
                    continue

            if not last_dates:
                lines.append("⚠️  Price Data: Could not parse date from sample files.")
            else:
                last_dates.sort()
                median_date = last_dates[len(last_dates) // 2]
                lag = _trading_days_old(median_date)
                status = "✅" if lag <= 1 else ("⚠️ " if lag <= 5 else "❌")
                lines.append(
                    f"{status} Price Data: {len(price_files)} tickers | "
                    f"Median last date: {median_date} "
                    f"({lag} trading day(s) old)"
                )

        fund_dirs = [d for d in config.FUNDAMENTALS_DIR.glob("*") if d.is_dir()]
        if not fund_dirs:
            lines.append("❌ Fundamentals: No data found.")
        else:
            fund_mtimes = []
            for fd in fund_dirs:
                files = list(fd.glob("*.csv")) + list(fd.glob("*.parquet"))
                if files:
                    fund_mtimes.append(
                        max(f.stat().st_mtime for f in files)
                    )
            if fund_mtimes:
                import time
                latest_mtime = max(fund_mtimes)
                age_days = int((time.time() - latest_mtime) / 86400)
                f_status = "✅" if age_days <= 7 else ("⚠️ " if age_days <= 30 else "❌")
                lines.append(
                    f"{f_status} Fundamentals: {len(fund_dirs)} tickers | "
                    f"Most recent update: {age_days} days ago"
                )
            else:
                lines.append(f"ℹ️  Fundamentals: {len(fund_dirs)} dirs found (no CSV/parquet files inside).")

        macro_files = list(config.ALTERNATIVE_DIR.glob("*.csv"))
        lines.append(f"ℹ️  Macro/Alt Data: {len(macro_files)} indicators available.")

        pred_dir = config.RESULTS_DIR / "predictions"
        pred_files = sorted(pred_dir.glob("alpha_signals_*.parquet"))
        
        if pred_files:
            latest_file = pred_files[-1]
            try:
                date_str = latest_file.stem.replace("alpha_signals_", "")
                last_signal = datetime.strptime(date_str, "%Y-%m-%d").date()
                signal_lag = _trading_days_old(last_signal)
                
                if signal_lag <= 1:
                    s_status = "✅"
                elif signal_lag <= 5:
                    s_status = "⚠️ "
                else:
                    s_status = "❌"
                    
                lines.append(
                    f"{s_status} Daily Signals: last generated {last_signal} "
                    f"({signal_lag} trading days ago)"
                )
            except ValueError:
                lines.append(f"⚠️  Daily Signals: Could not parse date from {latest_file.name}")
        else:
            lines.append("❌ Daily Signals: No prediction files found in results/predictions/")

        self.add_section("1. System Health & Data Integrity", "\n".join(lines))

    def analyze_factors(self):
        """
        Summarizes Alpha Factor efficacy and statistical significance.
        
        Aggregates factor performance and applies heuristic sorting. The primary
        ranking relies on Mean IC (Signal Strength) and subsequently the t-statistic.

        Args:
            None

        Returns:
            None
        """
        report_path = config.RESULTS_DIR / "validation" / "factor_validation_report.csv"
        if not report_path.exists():
            self.add_section(
                "2. Factor Quality Assurance",
                "⚠️  No factor validation report found. Run `validate_factors.py`."
            )
            return

        df = pd.read_csv(report_path)

        total    = len(df)
        passing  = df[df["status"].str.startswith("PASS")].shape[0]
        warnings = df[df["status"].str.startswith("WARN")].shape[0]
        failing  = total - passing - warnings

        lines = [
            f"Summary: {passing} PASS | {warnings} WARN | {failing} FAIL "
            f"(Total: {total} factors)"
        ]

        if "ic_mean" in df.columns:
            ic_std_col = "ic_std" if "ic_std" in df.columns else None
            if ic_std_col and "n_dates" in df.columns:
                df["_tstat"] = (
                    df["ic_mean"] /
                    (df[ic_std_col] / (df["n_dates"] ** 0.5).clip(lower=1))
                ).fillna(0)
                sort_col = "ic_mean"
            elif "icir" in df.columns:
                df["_tstat"] = df["icir"]
                sort_col = "ic_mean"
            else:
                df["_tstat"] = 0.0
                sort_col = "ic_mean"

            top_5 = df.sort_values(sort_col, ascending=False).head(5)
            lines.append("\nTop 5 Alpha Drivers (by Mean IC):")
            hdr = (f"| {'Factor':<30} | {'IC Mean':>8} | "
                   f"{'t-stat':>7} | {'Status':<12} |")
            lines.append(hdr)
            lines.append("|" + "-"*32 + "|" + "-"*10 + "|" + "-"*9 + "|" + "-"*14 + "|")
            for _, row in top_5.iterrows():
                lines.append(
                    f"| {row['factor']:<30} | "
                    f"{row['ic_mean']:>8.4f} | "
                    f"{row['_tstat']:>7.1f} | "
                    f"{row['status']:<12} |"
                )

        self.add_section("2. Factor Quality Assurance", "\n".join(lines))

    def analyze_models(self):
        """
        Evaluates Out-of-Sample (OOS) Machine Learning Model Performance.
        
        Extracts persistent metrics embedded within the pickled model artifacts 
        and assesses their tiering classification (PROD / ENSEMBLE / GATED).

        Args:
            None

        Returns:
            None
        """
        model_dir   = config.MODELS_DIR / "production"
        model_files = list(model_dir.glob("*_latest.pkl"))

        if not model_files:
            self.add_section(
                "3. Model Robustness",
                "⚠️  No production models found. Run `train_models.py`."
            )
            return

        lines = [f"Found {len(model_files)} production model(s)."]
        lines.append(
            "Note: t-stat = IC_mean / (IC_std/√N_days). "
            "AnnICIR = DailyICIR × √252. t > 3 = strong signal.\n"
        )

        hdr = (f"| {'Model':<12} | {'OOS IC':>8} | {'t-stat':>7} | "
               f"{'AnnICIR':>8} | {'Tier':<10} | {'Trained To':<12} |")
        lines.append(hdr)
        lines.append(
            "|" + "-"*14 + "|" + "-"*10 + "|" + "-"*9 + "|"
            + "-"*10 + "|" + "-"*12 + "|" + "-"*14 + "|"
        )

        for pkl in sorted(model_files):
            try:
                data       = joblib.load(pkl)
                name       = pkl.stem.replace("_latest", "").capitalize()
                m          = data.get("oos_metrics", {})
                trained_to = data.get("trained_to", "Unknown")

                if not m:
                    lines.append(f"| {name:<12} | {'⚠️  METRICS MISSING (Retrain Model)':<43} |")
                    continue

                ic   = m.get("ic_mean", 0.0)
                std  = m.get("ic_std",  1e-8)
                n_d  = m.get("n_dates", 1)
                tstat     = ic / (std / (n_d ** 0.5)) if n_d > 0 else 0.0
                ann_icir  = (ic / std) * (252 ** 0.5) if std > 0 else 0.0

                # Bind mathematical governance parameters mirroring the configuration file
                PROD_IC     = getattr(config, "PROD_IC_THRESHOLD",  0.010)
                PROD_TSTAT  = getattr(config, "PROD_IC_TSTAT",       2.5)
                ENS_IC      = getattr(config, "MIN_OOS_IC_THRESHOLD", 0.005)
                ENS_TSTAT   = getattr(config, "MIN_OOS_IC_TSTAT",     1.5)

                if ic >= PROD_IC and tstat >= PROD_TSTAT:
                    tier = "✅ PROD"
                elif ic >= ENS_IC and tstat >= ENS_TSTAT:
                    tier = "🟡 ENSEMBLE"
                else:
                    tier = "❌ GATED"

                lines.append(
                    f"| {name:<12} | {ic:>+8.4f} | {tstat:>7.1f} | "
                    f"{ann_icir:>8.2f} | {tier:<10} | {trained_to:<12} |"
                )
            except Exception as exc:
                lines.append(
                    f"| {pkl.stem:<12} | ERROR loading: {exc} |"
                )

        self.add_section("3. Model Robustness", "\n".join(lines))

    def analyze_backtests(self):
        """
        Compares Strategy Performance against the S&P 500 Benchmark.
        
        Aggregates risk and return metrics, computing excess returns and regime 
        sensitivity via annual performance breakdowns.

        Args:
            None

        Returns:
            None
        """
        results_dir  = config.RESULTS_DIR
        backtest_dirs = sorted(results_dir.glob("backtest_*"))

        if not backtest_dirs:
            self.add_section(
                "4. Strategy Performance (Backtest)",
                "⚠️  No backtest results found. Run `run_backtest.py` or `train_models.py --all`."
            )
            return

        lines      = []
        dd_details = []
        yr_details = []

        hdr = (f"| {'Method':<20} | {'CAGR':>8} | {'Sharpe':>6} | "
               f"{'Sortino':>7} | {'MaxDD':>8} | {'vs SPY':>8} | {'End Equity':>12} |")
        lines.append(hdr)
        lines.append(
            "|" + "-"*22 + "|" + "-"*10 + "|" + "-"*8 + "|"
            + "-"*9 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*14 + "|"
        )

        for d in backtest_dirs:
            method  = d.name.replace("backtest_", "")
            eq_path = d / "equity_curve.csv"
            if not eq_path.exists():
                continue
            try:
                eq = pd.read_csv(eq_path)
                eq["date"] = pd.to_datetime(eq["date"])
                eq = eq.set_index("date")["total_value"].sort_index()

                cagr    = _calc_cagr(eq)
                sharpe  = _calc_sharpe(eq)
                sortino = _calc_sortino(eq)
                maxdd   = _calc_max_dd(eq)
                end_val = eq.iloc[-1]
                s_date  = eq.index[0].strftime("%Y-%m-%d")
                e_date  = eq.index[-1].strftime("%Y-%m-%d")

                spy_ret = _fetch_spy(s_date, e_date)
                if spy_ret is not None:
                    strat_ret = eq.pct_change().dropna()
                    aligned   = pd.DataFrame(
                        {"s": strat_ret, "b": spy_ret}
                    ).dropna()
                    if not aligned.empty:
                        spy_cagr = _calc_cagr(
                            (1 + aligned["b"]).cumprod() * eq.iloc[0]
                        )
                        excess = cagr - spy_cagr
                        vs_spy_str = _format_pct(excess)
                    else:
                        vs_spy_str = "  N/A  "
                else:
                    vs_spy_str = "  N/A  "

                lines.append(
                    f"| {method:<20} | {_format_pct(cagr):>8} | "
                    f"{_format_float(sharpe):>6} | {_format_float(sortino):>7} | "
                    f"{_format_pct(maxdd):>8} | {vs_spy_str:>8} | "
                    f"{_format_currency(end_val):>12} |"
                )

                top_dds = _get_top_drawdowns(eq, n=3)
                if top_dds:
                    dd_details.append(
                        f"\n{method} — Top {len(top_dds)} Drawdown(s) "
                        f"({s_date} to {e_date}):"
                    )
                    dd_hdr = (f"| {'Depth':>8} | {'Start':<10} | "
                              f"{'Valley':<10} | {'Recovery':<10} | {'Days':>5} |")
                    dd_details.append(dd_hdr)
                    dd_details.append(
                        "|" + "-"*10 + "|" + "-"*12 + "|"
                        + "-"*12 + "|" + "-"*12 + "|" + "-"*7 + "|"
                    )
                    for dd in top_dds:
                        rec_str = (
                            dd["recovery"].strftime("%Y-%m-%d")
                            if not pd.isna(dd["recovery"]) else "Active ⚠️"
                        )
                        dd_details.append(
                            f"| {_format_pct(dd['depth']):>8} | "
                            f"{dd['start'].strftime('%Y-%m-%d'):<10} | "
                            f"{dd['valley'].strftime('%Y-%m-%d'):<10} | "
                            f"{rec_str:<10} | {dd['days']:>5} |"
                        )

                rets = eq.pct_change().dropna()
                rets.index = pd.to_datetime(rets.index)
                spy_daily = spy_ret if spy_ret is not None else None

                yr_details.append(f"\n{method} — Annual Performance:")
                yr_hdr = (f"| {'Year':>4} | {'Strategy':>9} | "
                          f"{'S&P 500':>8} | {'Excess':>8} | {'Note':<6} |")
                yr_details.append(yr_hdr)
                yr_details.append(
                    "|" + "-"*6 + "|" + "-"*11 + "|"
                    + "-"*10 + "|" + "-"*10 + "|" + "-"*8 + "|"
                )

                for yr in sorted(rets.index.year.unique()):
                    yr_mask = rets.index.year == yr
                    yr_rets  = rets[yr_mask]
                    yr_cagr  = (1 + yr_rets).prod() ** (252 / len(yr_rets)) - 1

                    if spy_daily is not None:
                        spy_yr = spy_daily[spy_daily.index.year == yr]
                        if len(spy_yr) > 0:
                            spy_yr_cagr = (1 + spy_yr).prod() ** (252 / len(spy_yr)) - 1
                            excess_yr   = yr_cagr - spy_yr_cagr
                            icon        = "✅" if excess_yr >= 0 else "❌"
                            yr_details.append(
                                f"| {yr:>4} | {_format_pct(yr_cagr):>9} | "
                                f"{_format_pct(spy_yr_cagr):>8} | "
                                f"{_format_pct(excess_yr):>8} | {icon:<6} |"
                            )
                            continue
                    yr_details.append(
                        f"| {yr:>4} | {_format_pct(yr_cagr):>9} | "
                        f"{'N/A':>8} | {'N/A':>8} | {'':6} |"
                    )

            except Exception as exc:
                lines.append(
                    f"| {method:<20} | ERROR: {exc} |"
                )

        if dd_details:
            lines.append("\n" + "\n".join(dd_details))
        if yr_details:
            lines.append("\n" + "\n".join(yr_details))

        self.add_section("4. Strategy Performance (Backtest)", "\n".join(lines))

    def analyze_latest_orders(self):
        """
        Audits the current target portfolio construction logic.
        
        Verifies that orders are derived from timely inferences and assesses 
        concentration risk using the Herfindahl-Hirschman Index (HHI) alongside 
        gross/net exposure levels.

        Args:
            None

        Returns:
            None
        """
        order_path = config.RESULTS_DIR / "orders" / "orders_latest.csv"

        if not order_path.exists():
            self.add_section(
                "5. Current Positioning",
                "⚠️  No active orders found. Run `optimize_portfolio.py`."
            )
            return

        df = pd.read_csv(order_path)
        lines = []

        signal_date = None
        if "signal_date" in df.columns:
            signal_date = pd.to_datetime(df["signal_date"].iloc[0]).date()
        elif "date" in df.columns:
            signal_date = pd.to_datetime(df["date"].iloc[0]).date()

        if signal_date is not None:
            lag = _trading_days_old(signal_date)
            if lag is not None:
                if lag > 21:
                    lines.append(
                        f"🚨 SIGNAL AGE WARNING: Orders based on signals from "
                        f"{signal_date} ({lag} trading days ago). "
                        f"STRONGLY RECOMMEND regenerating signals before trading."
                    )
                elif lag > 5:
                    lines.append(
                        f"⚠️  Signal Age: {signal_date} ({lag} trading days ago). "
                        "Consider refreshing signals."
                    )
                else:
                    lines.append(
                        f"✅ Signal Date: {signal_date} ({lag} trading day(s) old)"
                    )

        longs  = df[df["side"] == "LONG"]  if "side" in df.columns else df
        shorts = df[df["side"] == "SHORT"] if "side" in df.columns else pd.DataFrame()

        total_val    = df["value"].sum() if "value" in df.columns else 0.0
        long_val     = longs["value"].sum() if "value" in longs.columns else 0.0
        short_val    = shorts["value"].sum() if "value" in shorts.columns else 0.0
        
        gross_exp = long_val + abs(short_val)
        net_exp   = long_val - abs(short_val)

        # Evaluate HHI concentration bounds to identify outsized allocations
        if "weight" in df.columns and df["weight"].abs().sum() > 0:
            w   = df["weight"].abs()
            w_norm = w / w.sum() 
            hhi = (w_norm ** 2).sum()
            hhi_str = f"{hhi:.4f}"
            hhi_label = "Concentrated ⚠️" if hhi > 0.10 else "Diversified ✅"
        else:
            hhi_str   = "N/A"
            hhi_label = ""

        lines.append(f"Net Position Val: {_format_currency(net_exp)}")
        lines.append(f"Long Value:       {_format_currency(long_val)}")
        lines.append(f"Short Value:      {_format_currency(short_val)}")
        lines.append(
            f"Positions:        {len(df)} "
            f"({len(longs)} Long | {len(shorts)} Short)"
        )
        lines.append(f"Gross Exposure:   {_format_currency(gross_exp)}")
        
        lines.append(f"HHI Concentration: {hhi_str}  {hhi_label}")

        if "weight" in df.columns:
            lines.append("\nTop 5 Positions:")
            top5 = df.sort_values("weight", ascending=False).head(5)
            hdr  = (f"| {'Ticker':<8} | {'Side':<6} | "
                    f"{'Weight':>8} | {'Value':>12} |")
            lines.append(hdr)
            lines.append("|" + "-"*10 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*14 + "|")
            for _, row in top5.iterrows():
                side = row.get("side", "LONG")
                val  = row.get("value", float("nan"))
                lines.append(
                    f"| {row['ticker']:<8} | {side:<6} | "
                    f"{_format_pct(row['weight']):>8} | "
                    f"{_format_currency(val):>12} |"
                )

        self.add_section("5. Current Positioning", "\n".join(lines))

    def print_report(self, output_file: str | None = None):
        """
        Renders the aggregated quantitative report to standard output or disk.

        Args:
            output_file (str | None, optional): Destination file path for the plain 
                text report. If None, the report is only printed to the console.

        Returns:
            None
        """
        output = [
            "=" * 70,
            "  QUANTITATIVE RESEARCH EXECUTIVE SUMMARY",
            f"  Generated: {self.report_date.strftime('%Y-%m-%d %H:%M')}",
            "=" * 70,
            "",
        ]

        for title, content in self.sections:
            output.append(f"{'─'*70}")
            output.append(f"  {title}")
            output.append(f"{'─'*70}")
            output.append(content)
            output.append("")

        final_text = "\n".join(output)
        print(final_text)

        if output_file:
            path = Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(final_text)
            logger.info(f"Report saved → {path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate Executive Quantitative Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_report.py
  python scripts/create_report.py --output-file results/executive_summary.txt
        """,
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Path to save report as plain text file."
    )
    args = parser.parse_args()

    report = QuantitativeManagerReport()

    logger.info("Compiling System Health...")
    report.check_data_health()

    logger.info("Analyzing Factors...")
    report.analyze_factors()

    logger.info("Analyzing Models...")
    report.analyze_models()

    logger.info("Analyzing Backtests...")
    report.analyze_backtests()

    logger.info("Analyzing Orders...")
    report.analyze_latest_orders()

    report.print_report(args.output_file)

if __name__ == "__main__":
    main()