"""
report.py
=========
Executive Quantitative Research Report
--------------------------------------
Aggregates insights from the entire alpha research pipeline into a single
management view. Focuses on system health, factor quality, model robustness,
and latest portfolio positioning.

Usage:
    python scripts/report.py
    python scripts/report.py --output-file results/executive_summary.md
"""

import sys
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config
from config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("Quant_Alpha")

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def _format_currency(x):
    if pd.isna(x): return "-"
    return f"${x:,.0f}"

def _format_pct(x):
    if pd.isna(x): return "-"
    return f"{x:.2%}"

def _format_float(x, prec=2):
    if pd.isna(x): return "-"
    return f"{x:.{prec}f}"

def _calc_cagr(series):
    if len(series) < 2: return 0.0
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    days = (series.index[-1] - series.index[0]).days
    if days <= 0: return 0.0
    return (1 + total_ret) ** (365.0 / days) - 1

def _calc_max_dd(series):
    if len(series) < 1: return 0.0
    peak = series.cummax()
    dd = (series / peak) - 1
    return dd.min()

def _calc_sharpe(series):
    # series is daily portfolio value
    if len(series) < 2: return 0.0
    rets = series.pct_change().dropna()
    if rets.std() == 0: return 0.0
    return (rets.mean() / rets.std()) * np.sqrt(252)

# ------------------------------------------------------------------------------
# REPORT GENERATOR
# ------------------------------------------------------------------------------
class QuantitativeManagerReport:
    def __init__(self):
        self.report_date = datetime.now()
        self.sections = []

    def add_section(self, title, content):
        self.sections.append((title, content))

    def check_data_health(self):
        """Assess freshness of prices and fundamentals."""
        lines = []
        
        # Prices
        price_files = list(config.PRICES_DIR.glob("*.csv"))
        if not price_files:
            lines.append("❌ Price Data: No files found.")
        else:
            # Check a sample
            sample = pd.read_csv(price_files[0])
            last_date = pd.to_datetime(sample["date"]).max().date()
            lag = (self.report_date.date() - last_date).days
            status = "✅" if lag <= 1 else ("⚠️" if lag <= 3 else "❌")
            lines.append(f"{status} Price Data: {len(price_files)} tickers | Last Date: {last_date} ({lag} days ago)")

        # Fundamentals
        fund_dirs = [d for d in config.FUNDAMENTALS_DIR.glob("*") if d.is_dir()]
        if not fund_dirs:
            lines.append("❌ Fundamentals: No data found.")
        else:
            lines.append(f"ℹ️  Fundamentals: {len(fund_dirs)} tickers covered.")

        # Macro
        macro_files = list(config.ALTERNATIVE_DIR.glob("*.csv"))
        lines.append(f"ℹ️  Macro/Alt Data: {len(macro_files)} indicators available.")

        self.add_section("1. System Health & Data Integrity", "\n".join(lines))

    def analyze_factors(self):
        """Summarize factor validation results."""
        report_path = config.RESULTS_DIR / "validation" / "factor_validation_report.csv"
        if not report_path.exists():
            self.add_section("2. Factor Quality Assurance", "⚠️ No factor validation report found. Run `validate_factors.py`.")
            return

        df = pd.read_csv(report_path)
        
        # Summary Stats
        total = len(df)
        passing = df[df["status"].str.startswith("PASS")].shape[0]
        warnings = df[df["status"].str.startswith("WARN")].shape[0]
        failing = total - passing - warnings
        
        lines = []
        lines.append(f"**Summary**: {passing} PASS | {warnings} WARN | {failing} FAIL (Total: {total})")
        
        # Top Factors
        if "icir" in df.columns:
            top_5 = df.sort_values("icir", ascending=False).head(5)
            lines.append("\n**Top 5 Alpha Drivers (by ICIR):**")
            header = f"| {'Factor':<25} | {'IC Mean':<10} | {'ICIR':<8} | {'Status':<15} |"
            lines.append(header)
            lines.append("|" + "-"*27 + "|" + "-"*12 + "|" + "-"*10 + "|" + "-"*17 + "|")
            for _, row in top_5.iterrows():
                lines.append(f"| {row['factor']:<25} | {row['ic_mean']:>10.4f} | {row['icir']:>8.2f} | {row['status']:<15} |")

        self.add_section("2. Factor Quality Assurance", "\n".join(lines))

    def analyze_models(self):
        """Summarize production model performance."""
        model_dir = config.MODELS_DIR / "production"
        model_files = list(model_dir.glob("*_latest.pkl"))
        
        if not model_files:
            self.add_section("3. Model Robustness", "⚠️ No production models found. Run `train_models.py`.")
            return

        lines = []
        lines.append(f"Found {len(model_files)} active production models.")
        
        header = f"| {'Model':<15} | {'OOS IC':<10} | {'OOS ICIR':<10} | {'Trained To':<12} |"
        lines.append(header)
        lines.append("|" + "-"*17 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*14 + "|")

        for pkl in model_files:
            try:
                data = joblib.load(pkl)
                name = pkl.stem.replace("_latest", "").capitalize()
                metrics = data.get("oos_metrics", {})
                trained_to = data.get("trained_to", "Unknown")
                
                ic = metrics.get("ic_mean", 0.0)
                icir = metrics.get("icir", 0.0)
                
                lines.append(f"| {name:<15} | {ic:>10.4f} | {icir:>10.2f} | {trained_to:<12} |")
            except Exception as e:
                lines.append(f"| {pkl.name:<15} | ERROR | - | - |")

        self.add_section("3. Model Robustness", "\n".join(lines))

    def analyze_backtests(self):
        """Summarize backtest performance across methods."""
        results_dir = config.RESULTS_DIR
        backtest_dirs = list(results_dir.glob("backtest_*"))
        
        if not backtest_dirs:
            self.add_section("4. Strategy Performance (Backtest)", "⚠️ No backtest results found.")
            return

        lines = []
        header = f"| {'Method':<20} | {'CAGR':<8} | {'Sharpe':<6} | {'MaxDD':<8} | {'End Equity':<12} |"
        lines.append(header)
        lines.append("|" + "-"*22 + "|" + "-"*10 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*14 + "|")

        for d in backtest_dirs:
            method = d.name.replace("backtest_", "")
            eq_path = d / "equity_curve.csv"
            
            if eq_path.exists():
                try:
                    eq = pd.read_csv(eq_path)
                    eq["date"] = pd.to_datetime(eq["date"])
                    eq = eq.set_index("date")["total_value"]
                    
                    cagr = _calc_cagr(eq)
                    sharpe = _calc_sharpe(eq)
                    maxdd = _calc_max_dd(eq)
                    end_val = eq.iloc[-1]
                    
                    lines.append(f"| {method:<20} | {_format_pct(cagr):<8} | {_format_float(sharpe):<6} | {_format_pct(maxdd):<8} | {_format_currency(end_val):<12} |")
                except Exception:
                    lines.append(f"| {method:<20} | ERROR | - | - | - |")

        self.add_section("4. Strategy Performance (Backtest)", "\n".join(lines))

    def analyze_latest_orders(self):
        """Summarize current portfolio positioning."""
        order_path = config.RESULTS_DIR / "orders" / "orders_latest.csv"
        
        if not order_path.exists():
            self.add_section("5. Current Positioning", "⚠️ No active orders found.")
            return

        df = pd.read_csv(order_path)
        total_val = df["value"].sum()
        longs = df[df["side"] == "LONG"]
        shorts = df[df["side"] == "SHORT"]
        
        lines = []
        lines.append(f"**Total Exposure**: {_format_currency(total_val)}")
        lines.append(f"**Positions**: {len(df)} ({len(longs)} Long | {len(shorts)} Short)")
        
        # Top 5 Positions
        lines.append("\n**Top 5 Positions:**")
        top_5 = df.sort_values("weight", ascending=False).head(5)
        
        header = f"| {'Ticker':<8} | {'Side':<6} | {'Weight':<8} | {'Value':<12} |"
        lines.append(header)
        lines.append("|" + "-"*10 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*14 + "|")
        
        for _, row in top_5.iterrows():
            lines.append(f"| {row['ticker']:<8} | {row['side']:<6} | {_format_pct(row['weight']):<8} | {_format_currency(row['value']):<12} |")

        self.add_section("5. Current Positioning", "\n".join(lines))

    def print_report(self, output_file=None):
        output = []
        output.append("="*60)
        output.append(f"QUANTITATIVE RESEARCH EXECUTIVE SUMMARY")
        output.append(f"Generated: {self.report_date.strftime('%Y-%m-%d %H:%M')}")
        output.append("="*60 + "\n")

        for title, content in self.sections:
            output.append(f"## {title}")
            output.append("-" * len(title))
            output.append(content)
            output.append("\n")

        final_text = "\n".join(output)
        
        print(final_text)
        
        if output_file:
            path = Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(final_text)
            logger.info(f"Report saved to {path}")

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Executive Quantitative Report")
    parser.add_argument("--output-file", type=str, default=None, help="Path to save report (e.g. report.md)")
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
