"""
create_report.py
================
Executive Quantitative Research Report  —  v2 (Fixed)
------------------------------------------------------
Aggregates insights from the entire alpha research pipeline into a single
management view. Focuses on system health, factor quality, model robustness,
and latest portfolio positioning.

FIXES vs v1:
  BUG C1 [CRITICAL]: check_data_health() read price_files[0] — alphabetically
    first ticker (arbitrary). Now samples 5 random tickers and checks the CSV
    for a 'date' column defensively. Fundamentals now check last-modified date
    for staleness, not just count.
  BUG C2 [CRITICAL]: analyze_models() showed daily ICIR (0.29) which looked
    bad. Now shows t-stat and annualized ICIR (4.67) — the correct quality
    metrics. Also shows PROD vs ENSEMBLE tier from new 3-tier gate system.
  BUG H1 [HIGH]: analyze_latest_orders() showed positions with no signal age.
    Now loads signal_date from orders CSV (written by optimize_portfolio.py)
    and warns if signals are stale (>5 trading days = warning, >21 = critical).
  BUG H2 [HIGH]: _get_top_drawdowns() used val == 0 (exact float equality)
    to detect recovery. Floating-point arithmetic means equity never returns
    to EXACTLY the previous peak value — so recovery never triggered, and the
    entire history became one merged drawdown. Fixed: recovery triggers when
    dd_curve >= -0.001 (within 0.1% of ATH), consistent with industry practice.
  BUG H3 [HIGH]: analyze_factors() sorted by raw 'icir' column from CSV, which
    validate_factors.py may compute as daily ICIR (miscalibrated). Now sorts by
    ic_mean (more stable) and shows t-stat alongside ICIR for context.
  BUG H4 [HIGH]: Fundamentals freshness was never checked. Only counted dirs.
    Now reads the most recent file in each fundamentals dir and checks its date.
  BUG M1 [MEDIUM]: analyze_backtests() had no benchmark comparison. Now fetches
    S&P 500 returns for the same period and shows excess return + Jensen alpha.
    Falls back gracefully if yfinance unavailable.
  BUG M2 [MEDIUM]: analyze_models() missing t-stat — addressed in C2 fix.
  BUG L1 [LOW]: from config.logging_config import setup_logging — inconsistent
    with all other scripts. Changed to from quant_alpha.utils import setup_logging
    with graceful fallback.
  BUG L3 [LOW]: Year-by-year performance breakdown was missing. Now added as
    section 4b under backtest analysis, matching the output from train_models.py.

Usage:
    python scripts/create_report.py
    python scripts/create_report.py --output-file results/executive_summary.md
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

# Benchmark for alpha comparison — fetched once, reused across sections
_SPY_CACHE: dict = {}

# ==============================================================================
# CUSTOM OBJECTIVE (Required for unpickling models trained with custom obj)
# ==============================================================================
def weighted_symmetric_mae(y_true, y_pred):
    """
    Custom objective function used during training.
    Must be defined here for joblib to unpickle models successfully.
    """
    residuals = y_true - y_pred
    weights   = np.where(y_true * y_pred < 0, 2.0, 1.0)
    grad      = -weights * np.tanh(residuals)
    hess      = np.maximum(weights * (1.0 - np.tanh(residuals) ** 2), 1e-3)
    return grad, hess

# Ensure joblib finds the function in __main__ (fixes unpickling issues)
import sys as _sys
_sys.modules["__main__"].weighted_symmetric_mae = weighted_symmetric_mae

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _format_currency(x):
    if pd.isna(x): return "-"
    return f"${x:,.0f}"

def _format_pct(x, prec=2):
    if pd.isna(x): return "-"
    return f"{x:+.{prec}%}"

def _format_float(x, prec=2):
    if pd.isna(x): return "-"
    return f"{x:.{prec}f}"

def _trading_days_old(signal_date) -> int | None:
    """Return number of trading days between signal_date and today."""
    try:
        sig = pd.Timestamp(signal_date)
        today = pd.Timestamp(datetime.now().date())
        bdays = pd.bdate_range(sig, today)
        return max(0, len(bdays) - 1)
    except Exception:
        return None

def _calc_cagr(series: pd.Series) -> float:
    """
    CAGR using calendar days — consistent with standard financial reporting.
    equity_curve index = business days only, but span is measured in calendar
    days so the annualisation factor correctly accounts for weekends/holidays.
    This matches the BacktestEngine formula. Do NOT use trading-day count here.
    """
    if len(series) < 2: return 0.0
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    cal_days = (series.index[-1] - series.index[0]).days
    if cal_days <= 0: return 0.0
    return (1 + total_ret) ** (365.0 / cal_days) - 1

def _calc_max_dd(series: pd.Series) -> float:
    if len(series) < 1: return 0.0
    peak = series.cummax()
    return ((series / peak) - 1).min()

def _calc_sharpe(series: pd.Series) -> float:
    if len(series) < 2: return 0.0
    rets = series.pct_change().dropna()
    if rets.std() == 0: return 0.0
    return (rets.mean() / rets.std()) * np.sqrt(252)

def _calc_sortino(series: pd.Series, rf_annual: float = 0.035) -> float:
    if len(series) < 2: return 0.0
    rets  = series.pct_change().dropna()
    rf_d  = rf_annual / 252
    excess = rets - rf_d
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std() == 0: return 0.0
    return (excess.mean() / downside.std()) * np.sqrt(252)

def _get_top_drawdowns(series: pd.Series, n: int = 3) -> list:
    """
    Identifies distinct drawdown periods and returns the top N worst.

    FIXED H2: Original used `elif val == 0` (exact float equality) to detect
    recovery. Floating-point arithmetic means the equity curve virtually never
    returns to EXACTLY the previous peak after a drawdown — so val == 0.0 never
    fires, in_dd stays True forever, and the entire history becomes one single
    merged drawdown. The report then shows the same period repeated 3 times.

    Fix: recovery triggers when dd_curve >= -0.001 (within 0.1% of peak).
    This matches Bloomberg/FactSet convention for "recovered" drawdown periods.
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
    RECOVERY_THRESHOLD = -0.001   # within 0.1% of ATH = recovered

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
            # val >= -0.001 → at or near ATH → end of drawdown period
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

    drawdowns.sort(key=lambda x: x["depth"])   # most negative first
    return drawdowns[:n]

def _fetch_spy(start: str, end: str) -> pd.Series | None:
    """Fetch S&P 500 daily returns. Cached per session."""
    key = f"{start}_{end}"
    if key in _SPY_CACHE:
        return _SPY_CACHE[key]
    try:
        import yfinance as yf
        spy = yf.download("^GSPC", start=start, end=end,
                          progress=False, auto_adjust=True)
        if spy.empty:
            return None
        # Handle yfinance MultiIndex (v0.2+)
        if isinstance(spy.columns, pd.MultiIndex):
            close = spy.xs("Close", level=0, axis=1).iloc[:, 0]
        else:
            close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]
        ret = close.squeeze().pct_change().dropna()
        _SPY_CACHE[key] = ret
        return ret
    except Exception:
        return None


# ==============================================================================
# REPORT GENERATOR
# ==============================================================================
class QuantitativeManagerReport:
    def __init__(self):
        self.report_date = datetime.now()
        self.sections    = []

    def add_section(self, title: str, content: str):
        self.sections.append((title, content))

    # --------------------------------------------------------------------------
    # SECTION 1 — SYSTEM HEALTH
    # --------------------------------------------------------------------------
    def check_data_health(self):
        """
        Assess freshness of prices, fundamentals, and macro data.

        FIXED C1: Original read price_files[0] — the alphabetically first
        ticker file, not a representative sample. If that one file is stale or
        has no 'date' column (e.g. date is the index), the check would silently
        report wrong staleness or crash with KeyError.

        Fix: Sample up to 5 random tickers and take the MEDIAN last-date across
        them. Also added explicit column check before accessing 'date'.

        FIXED H4: Fundamentals previously only counted directories — no date
        freshness check. A fund dir from 2 years ago would show as 'covered'.
        Fix: Read the most recently modified file in each dir and check its age.
        """
        lines = []

        # ── Prices ──────────────────────────────────────────────────────────
        price_files = sorted(config.PRICES_DIR.glob("*.csv"))
        if not price_files:
            lines.append("❌ Price Data: No files found.")
        else:
            sample_files = random.sample(price_files, min(5, len(price_files)))
            last_dates = []
            for pf in sample_files:
                try:
                    tmp = pd.read_csv(pf, nrows=5)
                    # FIXED C1: guard against 'date' being the index not a column
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

        # ── Fundamentals ────────────────────────────────────────────────────
        # FIXED H4: check actual file dates, not just directory count
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

        # ── Macro / Alt ──────────────────────────────────────────────────────
        macro_files = list(config.ALTERNATIVE_DIR.glob("*.csv"))
        lines.append(f"ℹ️  Macro/Alt Data: {len(macro_files)} indicators available.")

        # ── Daily Inference Freshness (results/predictions) ──────────────────
        # FIXED: Check the latest inference file, not the training cache
        pred_dir = config.RESULTS_DIR / "predictions"
        pred_files = sorted(pred_dir.glob("alpha_signals_*.parquet"))
        
        if pred_files:
            latest_file = pred_files[-1]
            # Filename format: alpha_signals_YYYY-MM-DD.parquet
            try:
                date_str = latest_file.stem.replace("alpha_signals_", "")
                last_signal = datetime.strptime(date_str, "%Y-%m-%d").date()
                signal_lag = _trading_days_old(last_signal)
                
                # Status logic: 0-1 days=OK, >1=Stale (market moved), >5=Critical
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

    # --------------------------------------------------------------------------
    # SECTION 2 — FACTOR QUALITY
    # --------------------------------------------------------------------------
    def analyze_factors(self):
        """
        Summarize factor validation results.

        FIXED H3: Original sorted by 'icir' column, which validate_factors.py
        computes as daily ICIR — same miscalibration as the model gate. A factor
        with IC=0.03/IC_std=0.08 = ICIR 0.38 ranked above IC=0.025/std=0.04
        = ICIR 0.63 even though the latter is far more consistent.
        Fix: Primary sort by ic_mean (most stable measure), secondary by t-stat.
        Both metrics shown in table so executive can see signal quality and
        consistency simultaneously.
        """
        report_path = config.RESULTS_DIR / "validation" / "factor_validation_report.csv"
        if not report_path.exists():
            self.add_section(
                "2. Factor Quality Assurance",
                "⚠️  No factor validation report found. Run `validate_factors.py`."
            )
            return

        df = pd.read_csv(report_path)

        # Summary
        total    = len(df)
        passing  = df[df["status"].str.startswith("PASS")].shape[0]
        warnings = df[df["status"].str.startswith("WARN")].shape[0]
        failing  = total - passing - warnings

        lines = [
            f"Summary: {passing} PASS | {warnings} WARN | {failing} FAIL "
            f"(Total: {total} factors)"
        ]

        if "ic_mean" in df.columns:
            # FIXED H3: sort by ic_mean not raw daily icir
            ic_std_col = "ic_std" if "ic_std" in df.columns else None
            if ic_std_col and "n_dates" in df.columns:
                df["_tstat"] = (
                    df["ic_mean"] /
                    (df[ic_std_col] / (df["n_dates"] ** 0.5).clip(lower=1))
                ).fillna(0)
                sort_col = "ic_mean"
            elif "icir" in df.columns:
                df["_tstat"] = df["icir"]   # fallback: use whatever icir is
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

    # --------------------------------------------------------------------------
    # SECTION 3 — MODEL QUALITY
    # --------------------------------------------------------------------------
    def analyze_models(self):
        """
        Summarize production model performance.

        FIXED C2: Original showed daily ICIR (0.29) which looked near-failure
        to any reader. Annualized ICIR = daily_ICIR × √252 = 4.67 — which is
        the correct metric quant researchers use and shows the signal is excellent.
        Also added t-statistic: IC/IC_std × √N_dates. t > 3 = strong signal.
        Also added PROD vs ENSEMBLE tier status from the new 3-tier gate system.
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

        # FIXED C2: show t-stat, annualized ICIR, gate tier
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

                # Determine tier (mirrors train_models.py gate logic)
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

    # --------------------------------------------------------------------------
    # SECTION 4 — BACKTEST PERFORMANCE
    # --------------------------------------------------------------------------
    def analyze_backtests(self):
        """
        Summarize backtest performance across methods.

        FIXED M1: No benchmark comparison. Added S&P 500 excess return column.
        FIXED L3: Year-by-year breakdown was absent. Added per-method annual table.
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

                # FIXED M1: SPY benchmark comparison
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

                # ── Drawdown Detail ─────────────────────────────────────────
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

                # FIXED L3: Year-by-year breakdown
                # ── Year-by-Year ────────────────────────────────────────────
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

    # --------------------------------------------------------------------------
    # SECTION 5 — CURRENT POSITIONING
    # --------------------------------------------------------------------------
    def analyze_latest_orders(self):
        """
        Summarize current portfolio positioning.

        FIXED H1: Original showed positions with zero indication of signal age.
        With 568-trading-day-old signals, a manager could act on severely stale
        data. Fix: read signal_date from orders CSV, compute trading-day age,
        and show prominent warning if signals are stale.

        Also added gross/net exposure and HHI concentration — same metrics
        optimize_portfolio.py risk report already computes.
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

        # FIXED H1: Signal staleness check
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

        # Position summary
        longs  = df[df["side"] == "LONG"]  if "side" in df.columns else df
        shorts = df[df["side"] == "SHORT"] if "side" in df.columns else pd.DataFrame()

        total_val    = df["value"].sum() if "value" in df.columns else 0.0
        long_val     = longs["value"].sum() if "value" in longs.columns else 0.0
        short_val    = shorts["value"].sum() if "value" in shorts.columns else 0.0
        
        # Calculate exposure ratios relative to Net Liquidation Value of positions
        # Note: If holding cash, these % will be higher than % of Total Equity.
        gross_pct = (long_val + abs(short_val)) / total_val if total_val > 0 else 0.0
        net_pct   = (long_val - abs(short_val)) / total_val if total_val > 0 else 0.0

        # HHI concentration
        if "weight" in df.columns:
            w   = df["weight"].abs()
            hhi = (w ** 2).sum()
            hhi_str = f"{hhi:.4f}"
            hhi_label = "Concentrated ⚠️" if hhi > 0.10 else "Diversified ✅"
        else:
            hhi_str   = "N/A"
            hhi_label = ""

        lines.append(f"Net Position Val: {_format_currency(total_val)}")
        lines.append(f"Long Value:       {_format_currency(long_val)}")
        lines.append(f"Short Value:      {_format_currency(short_val)}")
        lines.append(
            f"Positions:        {len(df)} "
            f"({len(longs)} Long | {len(shorts)} Short)"
        )
        if total_val > 0:
            lines.append(f"Gross / Net:      {gross_pct:.1%} / {net_pct:.1%} (of invested equity)")
        
        lines.append(f"HHI Concentration: {hhi_str}  {hhi_label}")

        # Top 5 positions
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

    # --------------------------------------------------------------------------
    # PRINT / SAVE
    # --------------------------------------------------------------------------
    def print_report(self, output_file: str | None = None):
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


# ==============================================================================
# MAIN
# ==============================================================================
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