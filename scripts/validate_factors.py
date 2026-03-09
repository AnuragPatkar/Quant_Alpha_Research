"""
validate_factors.py
===================
Factor Quality Assurance & Validation Suite  — v2 (Fixed)
----------------------------------------------------------

Usage:
    python scripts/validate_factors.py
    python scripts/validate_factors.py --threshold 0.015 --top-n 20
    python scripts/validate_factors.py --force-rebuild
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend("Agg")  # headless server safe
from scipy import stats as scipy_stats
from tqdm import tqdm

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config
from quant_alpha.utils import setup_logging
from quant_alpha.utils import time_execution
from scripts.train_models import load_and_build_full_dataset, add_macro_features

setup_logging()
logger = logging.getLogger("Quant_Alpha")
warnings.filterwarnings("ignore")

# Metadata columns — never treated as factors
META_COLS = {
    "date", "ticker", "open", "high", "low", "close", "volume", "vwap",
    "sector", "industry", "target", "raw_ret_5d", "next_open", "future_open",
    "pnl_return", "index", "level_0", "split_factor", "div_factor",
    "macro_mom_5d", "macro_mom_21d", "macro_vix_proxy", "macro_trend_200d",
}

# IC decay lags to test (trading days)
IC_DECAY_LAGS = [1, 2, 3, 5, 10, 21]


class FactorValidator:
    def __init__(self, data: pd.DataFrame, target_col: str = "raw_ret_5d"):
        self.data       = data.sort_values(["ticker", "date"]).reset_index(drop=True)
        self.target_col = target_col
        self.factors    = self._identify_factors()
        self.results    = {}

        # Sanity check: target must be a FORWARD return, not contemporaneous
        self._assert_target_is_forward()

    # ──────────────────────────────────────────────────────────────────────────
    # SETUP
    # ──────────────────────────────────────────────────────────────────────────
    def _identify_factors(self) -> list[str]:
        return [
            c for c in self.data.columns
            if c not in META_COLS
            and not c.startswith("_")
            and pd.api.types.is_numeric_dtype(self.data[c])
        ]

    def _assert_target_is_forward(self):
        """
        Sanity check: target should NOT be highly correlated with today's
        intraday return. If it is, the shift is probably missing and IC will
        be meaningless (look-ahead bias).
        """
        if "close" not in self.data.columns or self.target_col not in self.data.columns:
            return
        intraday = (self.data["close"] / self.data["open"] - 1).dropna()
        target   = self.data[self.target_col].dropna()
        common   = intraday.index.intersection(target.index)
        if len(common) < 100:
            return
        corr = abs(intraday.loc[common].corr(target.loc[common]))
        if corr > 0.5:
            logger.warning(
                f"⚠️  target '{self.target_col}' correlates {corr:.2f} with "
                f"contemporaneous intraday return — check forward-shift!"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # 1. COVERAGE
    # ──────────────────────────────────────────────────────────────────────────
    @time_execution
    def check_coverage(self) -> pd.DataFrame:
        """Vectorized coverage stats — O(cols) not O(cols × rows)."""
        logger.info("🔍 Checking Data Coverage...")
        df = self.data[self.factors]
        n  = len(df)
        return pd.DataFrame({
            "coverage_pct": 1 - df.isna().mean(),
            "nan_count":    df.isna().sum(),
            "inf_count":    df.apply(lambda c: np.isinf(c).sum()),
            "zero_pct":     (df == 0).mean(),
        })

    # ──────────────────────────────────────────────────────────────────────────
    # 2. IC STATS  (raw + sector-neutral)
    # ──────────────────────────────────────────────────────────────────────────
    @time_execution
    def compute_ic_stats(self) -> pd.DataFrame:
        """
        Compute Rank IC (Spearman) and sector-neutral Rank IC per factor.

        Key fixes vs original:
          - Target ranks cached ONCE (not recomputed per factor)
          - Sector-neutral IC added (demean by date×sector before ranking)
          - Correct t-stat: t = ICIR × sqrt(N)
          - BH multiple-testing correction
          - pos_ic_pct threshold added to PASS gate
        """
        if "ic_stats" in self.results:
            return self.results["ic_stats"]

        logger.info(f"📉 Computing IC (raw + sector-neutral) | Target: {self.target_col}")

        valid      = self.data.dropna(subset=[self.target_col])
        has_sector = "sector" in valid.columns

        # Cache target ranks ONCE — big speedup vs original (recomputed per factor)
        t_ranks_raw = valid.groupby("date")[self.target_col].rank(pct=True)

        if has_sector:
            t_neutral = (valid.groupby(["date", "sector"])[self.target_col]
                         .transform(lambda x: (x - x.mean()) / (x.std() + 1e-8)))
        else:
            t_neutral = None

        ic_stats = []

        for f in tqdm(self.factors, desc="IC"):
            try:
                col = valid[f]
                if col.isna().mean() > 0.5:
                    continue  # skip sparse factors

                # ── Raw Rank IC ───────────────────────────────────────────────
                f_ranks  = valid.groupby("date")[f].rank(pct=True)
                tmp      = pd.DataFrame({"f": f_ranks, "t": t_ranks_raw,
                                         "date": valid["date"]})
                daily_ic = tmp.groupby("date").apply(
                    lambda x: x["f"].corr(x["t"]) if x["f"].std() > 1e-8 and x["t"].std() > 1e-8 else np.nan
                ).dropna()

                if len(daily_ic) < 3:
                    continue  # too few dates — t-stat would be meaningless

                ic_mean = float(daily_ic.mean())
                ic_std  = float(daily_ic.std()) + 1e-8
                icir    = ic_mean / ic_std
                n_dates = len(daily_ic)
                # FIXED t-stat: t = ICIR × sqrt(N)  (original ignored IC_std)
                t_stat  = icir * np.sqrt(n_dates)

                # ── Sector-neutral Rank IC ────────────────────────────────────
                sn_icir = np.nan
                if has_sector:
                    f_neutral    = (valid.groupby(["date", "sector"])[f]
                                    .transform(lambda x: (x - x.mean()) / (x.std() + 1e-8)))
                    sn_df        = pd.DataFrame({"f": f_neutral, "t": t_neutral,
                                                  "date": valid["date"]}).dropna()
                    sn_daily     = sn_df.groupby("date").apply(
                        lambda x: x["f"].corr(x["t"]) if x["f"].std() > 1e-8 and x["t"].std() > 1e-8 else np.nan
                    ).dropna()
                    sn_mean = float(sn_daily.mean())
                    sn_std  = float(sn_daily.std()) + 1e-8
                    sn_icir = sn_mean / sn_std

                ic_stats.append({
                    "factor":       f,
                    "ic_mean":      round(ic_mean, 5),
                    "ic_std":       round(ic_std, 5),
                    "icir":         round(icir, 4),
                    "t_stat":       round(t_stat, 3),
                    "pos_ic_pct":   round(float((daily_ic > 0).mean()), 3),
                    "n_dates":      n_dates,
                    "sn_icir":      round(sn_icir, 4) if not np.isnan(sn_icir) else np.nan,
                })

            except Exception as exc:
                logger.warning(f"IC failed for {f}: {exc}")

        if not ic_stats:
            return pd.DataFrame()

        df = pd.DataFrame(ic_stats).set_index("factor")

        # ── Multiple-testing correction (Benjamini-Hochberg FDR) ─────────────
        # Original: no correction → spurious significant factors
        from scipy.stats import t as t_dist
        try:
            from statsmodels.stats.multitest import multipletests
            n_d     = df["n_dates"].fillna(500).astype(int)
            p_vals  = df["t_stat"].apply(
                lambda t: float(2 * (1 - t_dist.cdf(abs(t), df=500)))
            ).values
            _, p_corr, _, _ = multipletests(p_vals, method="fdr_bh")
            df["p_value"]   = p_vals.round(4)
            df["p_bh_corr"] = p_corr.round(4)
        except ImportError:
            logger.warning("statsmodels not installed — skipping BH correction.")
            df["p_value"]   = np.nan
            df["p_bh_corr"] = np.nan

        df = df.sort_values("icir", ascending=False)
        self.results["ic_stats"] = df
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # 3. IC DECAY  (tells you rebalance frequency)
    # ──────────────────────────────────────────────────────────────────────────
    @time_execution
    def compute_ic_decay(self, top_factors: list[str]) -> pd.DataFrame:
        """
        IC at lags 1, 2, 3, 5, 10, 21 trading days.

        Interpretation:
          Fast decay (IC collapses by lag 3) → daily rebalance needed (high t-cost)
          Slow decay (IC stable to lag 10+)  → weekly/monthly rebalance fine
        """
        logger.info(f"⏳ Computing IC decay for top {len(top_factors)} factors...")
        decay_rows = []

        # Pre-compute shifted targets ONCE per lag — O(lags) not O(factors x lags)
        # Original computed this inside double loop = 20 factors x 6 lags = 120 groupby ops
        shifted_targets = {
            lag: self.data.groupby("ticker")[self.target_col].shift(-lag)
            for lag in IC_DECAY_LAGS
        }

        for f in tqdm(top_factors, desc="IC decay"):
            row = {"factor": f}
            for lag in IC_DECAY_LAGS:
                try:
                    shifted_target = shifted_targets[lag]
                    tmp = pd.DataFrame({
                        "f":    self.data[f],
                        "t":    shifted_target,
                        "date": self.data["date"]
                    }).dropna()
                    if tmp.empty:
                        row[f"ic_lag{lag}"] = np.nan
                        continue
                    f_r = tmp.groupby("date")["f"].rank(pct=True)
                    t_r = tmp.groupby("date")["t"].rank(pct=True)
                    tmp2 = pd.DataFrame({"f": f_r, "t": t_r, "date": tmp["date"]})
                    ic   = tmp2.groupby("date").apply(
                        lambda x: x["f"].corr(x["t"]) if x["f"].std() > 1e-8 and x["t"].std() > 1e-8 else np.nan
                    ).dropna().mean()
                    row[f"ic_lag{lag}"] = round(float(ic), 5)
                except Exception:
                    row[f"ic_lag{lag}"] = np.nan
            decay_rows.append(row)

        return pd.DataFrame(decay_rows).set_index("factor")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. AUTOCORRELATION  (per-ticker AR(1), averaged)
    # ──────────────────────────────────────────────────────────────────────────
    @time_execution
    def compute_autocorrelation(self) -> pd.DataFrame:
        """
        Per-ticker AR(1), then average across tickers.

        Original used panel-level corr which mixes cross-sectional and
        time-series variation — produces inflated, misleading autocorrelation.
        """
        logger.info("🔄 Computing Factor Autocorrelation (per-ticker AR1)...")
        auto_corrs = []

        for f in tqdm(self.factors, desc="Autocorr"):
            try:
                def _ticker_autocorr(s: pd.Series) -> float:
                    # Constant series (std=0): quarterly fundamentals unchanged
                    # for many days → s.corr(s.shift(1)) returns NaN.
                    # A perfectly constant signal has AR(1) = 1.0 by definition.
                    if s.std() < 1e-8:
                        return 1.0
                    return float(s.corr(s.shift(1)))

                per_ticker = self.data.groupby("ticker")[f].apply(_ticker_autocorr)
                auto_corrs.append({
                    "factor":       f,
                    "autocorr":     round(float(per_ticker.mean()), 4),
                    "autocorr_std": round(float(per_ticker.std()), 4),
                })
            except Exception:
                auto_corrs.append({"factor": f, "autocorr": np.nan, "autocorr_std": np.nan})

        return pd.DataFrame(auto_corrs).set_index("factor")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. REDUNDANCY HEATMAP
    # ──────────────────────────────────────────────────────────────────────────
    @time_execution
    def check_redundancy(self, top_n_factors: list[str]) -> None:
        """
        Spearman correlation heatmap.
        Uses year-stratified sample to preserve time structure.
        Original used random sample which breaks temporal correlation structure.
        """
        logger.info("📊 Generating Correlation Heatmap (year-stratified sample)...")
        if not top_n_factors:
            return

        # Year-stratified sample — fixes original random sample issue
        sample_df = (
            self.data[top_n_factors + ["date"]]
            .assign(year=lambda x: pd.to_datetime(x["date"]).dt.year)
            .groupby("year", group_keys=False)
            .apply(lambda x: x.sample(min(5_000, len(x)), random_state=42))
            .drop(columns=["year", "date"])
        )

        corr_matrix = sample_df.corr(method="spearman")

        plt.figure(figsize=(max(10, len(top_n_factors)//2),
                            max(8,  len(top_n_factors)//2)))
        sns.heatmap(corr_matrix, annot=len(top_n_factors) <= 20,
                    fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    linewidths=0.3)
        plt.title(f"Factor Spearman Correlation (Top {len(top_n_factors)}, year-stratified)")
        plt.tight_layout()

        out_path = config.RESULTS_DIR / "validation" / "factor_correlation.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"✅ Heatmap saved → {out_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. MASTER REPORT
    # ──────────────────────────────────────────────────────────────────────────
    def generate_report(self, output_dir: Path, threshold: float = 0.015,
                        top_n: int = 20):
        """Compile all metrics → master CSV + console summary."""
        coverage = self.check_coverage()
        ic_stats = self.compute_ic_stats()
        autocorr = self.compute_autocorrelation()

        if ic_stats.empty:
            logger.error("No IC stats computed — aborting report.")
            return

        report = ic_stats.join(coverage, how="left").join(autocorr, how="left")

        # PASS gate — stricter than original:
        #   |IC| > threshold  AND  coverage > 90%  AND  pos_ic_pct > 55%
        #   (original ignored pos_ic_pct — a factor with IC=0.02 on 3 lucky days
        #    would pass; now it needs to be consistent across 55%+ of days)
        # ── Classify every factor into one of 5 buckets ──────────────────
        report["status"]      = "FAIL (Weak Signal)"
        report["rescuable"]   = False

        # 1. PASS: strong, consistent, directional signal
        mask_pass = (
            (report["ic_mean"].abs() >= threshold) &
            (report["coverage_pct"]  > 0.90) &
            (report["pos_ic_pct"]    > 0.55)
        )
        report.loc[mask_pass, "status"] = "PASS"

        # 2. PASS (Inverted): negative IC but strong magnitude
        #    → multiply factor by -1 before use, signal is valid
        mask_inverted = (
            (report["ic_mean"]       <= -threshold) &
            (report["coverage_pct"]  >   0.90) &
            (report["pos_ic_pct"]    <   0.50)        # more negative IC days than positive
        )
        report.loc[mask_inverted, "status"]    = "PASS (Inverted — flip sign)"
        report.loc[mask_inverted, "rescuable"] = True

        # 3. WARN: high turnover (good IC but autocorr < 0.3 → expensive)
        mask_warn_turnover = (
            report["status"].str.startswith("PASS") &
            (report["autocorr"] < 0.3)
        )
        report.loc[mask_warn_turnover, "status"] = "WARN (High Turnover)"

        # 4. WARN: sector bias (raw IC >> sector-neutral ICIR)
        if "sn_icir" in report.columns:
            mask_sector_bias = (
                report["status"].str.startswith("PASS") &
                report["sn_icir"].notna() &
                (report["sn_icir"].abs() < report["icir"].abs() * 0.5)
            )
            report.loc[mask_sector_bias, "status"] = "WARN (Sector Bias)"

        # 5. RESCUABLE: low coverage but strong IC — fix data pipeline first
        mask_rescue_cov = (
            (report["ic_mean"].abs() >= threshold) &
            (report["coverage_pct"]  <= 0.90) &
            (report["coverage_pct"]  >  0.50)
        )
        report.loc[mask_rescue_cov, "status"]    = "FAIL (Fix Coverage)"
        report.loc[mask_rescue_cov, "rescuable"] = True

        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "factor_validation_report.csv"
        report.to_csv(csv_path)

        # ── IC Decay for top passing factors ──────────────────────────────────
        passing = report[report["status"].str.startswith("PASS")].head(top_n).index.tolist()
        decay_df = pd.DataFrame()
        if passing:
            decay_df = self.compute_ic_decay(passing)
            decay_df.to_csv(output_dir / "ic_decay.csv")

        # ── Console Summary ───────────────────────────────────────────────────
        n_pass_clean = (report["status"] == "PASS").sum()
        n_inverted   = (report["status"] == "PASS (Inverted — flip sign)").sum()
        n_warn       = report["status"].str.startswith("WARN").sum()
        n_fix_cov    = (report["status"] == "FAIL (Fix Coverage)").sum()
        n_weak       = (report["status"] == "FAIL (Weak Signal)").sum()
        n_usable     = n_pass_clean + n_inverted + n_warn

        sep  = "=" * 80
        sep2 = "-" * 37
        print("\n" + sep)
        print(f"  FACTOR VALIDATION REPORT  |  IC threshold: {threshold}")
        print(sep)
        print(f"  Total factors        : {len(report)}")
        print("  " + sep2)
        print(f"  PASS              : {n_pass_clean:<4} (use as-is)")
        print(f"  PASS (Inverted)   : {n_inverted:<4} (flip sign -> usable)")
        print(f"  WARN              : {n_warn:<4} (usable, check cost/bias)")
        print("  " + sep2)
        print(f"  Total Usable      : {n_usable:<4}")
        print("  " + sep2)
        print(f"  FAIL (Fix Cov)    : {n_fix_cov:<4} (potential, fix data)")
        print(f"  FAIL (Weak)       : {n_weak:<4} (drop)")

        cols = ["ic_mean", "icir", "sn_icir", "t_stat", "p_bh_corr",
                "autocorr", "coverage_pct", "pos_ic_pct", "status"]
        cols = [c for c in cols if c in report.columns]

        print(f"\n🏆 TOP 10 ALPHA DRIVERS:")
        print(report.sort_values("icir", ascending=False).head(10)[cols].to_string())

        if not decay_df.empty:
            print(f"\n⏳ IC DECAY (top {len(passing)} factors):")
            print(decay_df.to_string())
            print("  → Fast decay (lag5 ≈ 0): daily rebalance needed")
            print("  → Slow decay (lag10 still strong): weekly rebalance fine")

        failed = report[report["status"] == "FAIL"]
        print(f"\n❌ FAILED FACTORS ({len(failed)} total), bottom 5:")
        if not failed.empty:
            print(failed.sort_values("ic_mean").head(5)[["ic_mean","coverage_pct"]].to_string())

        print(f"\n💾 Full report → {csv_path}")

        # ── Redundancy heatmap ────────────────────────────────────────────────
        self.check_redundancy(passing[:top_n])

        return report


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Quant Alpha Factor Validation Suite v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python scripts/validate_factors.py
  python scripts/validate_factors.py --threshold 0.02 --top-n 25
  python scripts/validate_factors.py --force-rebuild
        """
    )
    parser.add_argument("--threshold",     type=float, default=0.015,
                        help="Minimum |IC| for PASS status (default: 0.015).")
    parser.add_argument("--top-n",         type=int,   default=20,
                        help="Top N passing factors to include in decay + heatmap.")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Rebuild dataset from raw files.")
    args = parser.parse_args()

    # 1. Load data (same loader as training — validates exactly what goes into model)
    logger.info("🚀 Loading Master Dataset...")
    try:
        data = load_and_build_full_dataset(force_rebuild=args.force_rebuild)
        data = add_macro_features(data)

        if "raw_ret_5d" not in data.columns:
            logger.info("⚙️  Constructing raw_ret_5d target...")
            data = data.sort_values(["ticker", "date"])
            next_open   = data.groupby("ticker")["open"].shift(-1).replace(0, np.nan)
            future_open = data.groupby("ticker")["open"].shift(-6)
            data["raw_ret_5d"] = (future_open / next_open) - 1

        data = data.dropna(subset=["raw_ret_5d"])

    except Exception as exc:
        logger.error(f"Data load failed: {exc}")
        sys.exit(1)

    logger.info(f"✅ Loaded: {len(data):,} rows × {len(data.columns)} cols")

    # 2. Validate
    validator = FactorValidator(data, target_col="raw_ret_5d")
    if not validator.factors:
        logger.error("❌ No numeric factors found!")
        sys.exit(1)

    print(f"\n🔍 Analysing {len(validator.factors)} factors...")
    output_dir = config.RESULTS_DIR / "validation"

    # 3. Report — ic_stats cached inside, so no double-compute
    report = validator.generate_report(output_dir,
                                       threshold=args.threshold,
                                       top_n=args.top_n)

    # 4. Cumulative IC plot for single best factor
    ic_stats = validator.results.get("ic_stats", pd.DataFrame())  # no second call
    if ic_stats.empty:
        return

    best = ic_stats.index[0]
    logger.info(f"📈 Plotting cumulative IC for best factor: {best}")

    valid = data.dropna(subset=[best, "raw_ret_5d"])
    # Rank per date, then correlate — this IS Spearman (Pearson on ranks)
    f_r = valid.groupby("date")[best].rank(pct=True)
    t_r = valid.groupby("date")["raw_ret_5d"].rank(pct=True)
    tmp = pd.DataFrame({"f": f_r, "t": t_r, "date": valid["date"]})
    daily_ic = tmp.groupby("date").apply(
        lambda x: x["f"].corr(x["t"]) if x["f"].std() > 1e-8 and x["t"].std() > 1e-8 else np.nan
    ).dropna()

    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    daily_ic.cumsum().plot(ax=ax1)
    ax1.set_title(f"Cumulative Rank IC: {best}")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(1, 2, 2)
    daily_ic.rolling(63).mean().plot(ax=ax2)
    ax2.set_title(f"Rolling 63-day IC: {best}")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.axhline(args.threshold, color="green", linestyle=":", linewidth=1,
                label=f"threshold={args.threshold}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"cum_ic_{best}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"✅ Saved → {out}")


if __name__ == "__main__":
    main()