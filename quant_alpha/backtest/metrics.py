"""
quant_alpha/backtest/metrics.py
================================
Performance metric calculations and report printing for backtest results.

Provides:
  - compute_metrics()         : full suite of annualised performance statistics
  - print_metrics_report()    : formatted stdout summary (used by run_backtest.py)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    equity_curve: pd.DataFrame,
    risk_free_rate: float = 0.035,
    trading_days: int = 252,
) -> Dict[str, Any]:
    """
    Compute a full suite of annualised performance statistics from an equity curve.

    Parameters
    ----------
    equity_curve  : DataFrame with columns ['date', 'total_value'].
    risk_free_rate: Annual risk-free rate (default 3.5%).
    trading_days  : Trading days per year (default 252).

    Returns
    -------
    Dict of metric name → value.
    """
    ec = equity_curve.copy()
    ec["date"] = pd.to_datetime(ec["date"])
    ec = ec.sort_values("date").reset_index(drop=True)

    nav    = ec["total_value"].values.astype(float)
    dates  = ec["date"]
    n_days = len(nav)

    if n_days < 2:
        return {"error": "Insufficient data for metric computation."}

    # ---- Daily returns ----
    daily_ret = np.diff(nav) / nav[:-1]
    daily_ret = daily_ret[np.isfinite(daily_ret)]

    if len(daily_ret) == 0:
        return {"error": "No finite daily returns."}

    # ---- CAGR ----
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr  = (nav[-1] / nav[0]) ** (1.0 / max(years, 1e-6)) - 1.0 if years > 0 else 0.0

    # ---- Volatility (annualised) ----
    ann_vol = float(np.std(daily_ret) * np.sqrt(trading_days))

    # ---- Sharpe ----
    rf_daily      = risk_free_rate / trading_days
    excess_daily  = daily_ret - rf_daily
    sharpe        = (
        float(np.mean(excess_daily) / np.std(excess_daily) * np.sqrt(trading_days))
        if np.std(excess_daily) > 1e-12
        else 0.0
    )

    # ---- Sortino ----
    downside = excess_daily[excess_daily < 0]
    downside_std = float(np.std(downside) * np.sqrt(trading_days)) if len(downside) > 1 else 1e-12
    sortino = float(np.mean(excess_daily) * trading_days / downside_std) if downside_std > 1e-12 else 0.0

    # ---- Max Drawdown ----
    hwm = np.maximum.accumulate(nav)
    safe_hwm = np.where(hwm > 1e-12, hwm, 1e-12)
    drawdowns = (nav - hwm) / safe_hwm
    max_dd    = float(abs(drawdowns.min())) if len(drawdowns) > 0 else 0.0

    # ---- Calmar ----
    calmar = float(cagr / max_dd) if max_dd > 1e-12 else 0.0

    # ---- Hit rate ----
    hit_rate = float((daily_ret > 0).mean())

    # ---- Total return ----
    total_return = float(nav[-1] / nav[0] - 1.0)

    return {
        "Total Return":      round(total_return, 6),
        "CAGR":              round(cagr, 6),
        "Ann. Volatility":   round(ann_vol, 6),
        "Sharpe Ratio":      round(sharpe, 4),
        "Sortino Ratio":     round(sortino, 4),
        "Max Drawdown":      round(max_dd, 6),
        "Calmar Ratio":      round(calmar, 4),
        "Hit Rate":          round(hit_rate, 4),
        "Start Date":        str(dates.iloc[0].date()),
        "End Date":          str(dates.iloc[-1].date()),
        "N Trading Days":    n_days,
    }


# ---------------------------------------------------------------------------
# Print helper — called directly by run_backtest.py
# ---------------------------------------------------------------------------

def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted performance metrics table to stdout.

    Parameters
    ----------
    metrics : Dict returned by compute_metrics() or engine.run()['metrics'].
    """
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  BACKTEST PERFORMANCE METRICS")
    print(f"{sep}")

    # Formatting rules per metric type
    pct_keys   = {"Total Return", "CAGR", "Ann. Volatility", "Max Drawdown", "Hit Rate"}
    ratio_keys = {"Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"}

    for k, v in metrics.items():
        if isinstance(v, float):
            if k in pct_keys:
                print(f"  {k:<30} {v:>10.2%}")
            elif k in ratio_keys:
                print(f"  {k:<30} {v:>10.4f}")
            else:
                print(f"  {k:<30} {v:>10.4f}")
        else:
            print(f"  {k:<30} {str(v):>10}")

    print(f"{sep}\n")