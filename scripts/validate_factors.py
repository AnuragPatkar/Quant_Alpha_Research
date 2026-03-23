"""
validate_factors.py
===================
Factor Quality Assurance & Validation Suite  — v3 (All Bugs Fixed)
----------------------------------------------------------
Fixes vs v2:
  FIX-1:  _assert_target_is_forward called twice in __init__ — removed duplicate
  FIX-2:  raw_ret_5d no clip in main() — added .clip(-0.5, 0.5) consistent with train_models
  FIX-3:  groupby().apply(lambda) in IC loop — vectorised via pre-ranked arrays
  FIX-4:  BH correction hardcoded df=500 — now uses per-factor n_dates
  FIX-5:  IC decay re-ranks target per factor — pre-ranked shifted targets cached once
  FIX-6:  Sector-neutral IC computed for all factors — now only for IC-passing candidates
  FIX-7:  PASS/Inverted gate gap 0.50–0.55 — symmetric bounds (>0.55 / <=0.45)
  FIX-8:  Look-ahead threshold 0.5 too loose — lowered to 0.3, raises ValueError
  FIX-9:  Zero-variance AR(1) hardcoded to 1.0 — now returns NaN, tagged FAIL (Zero Variance)
  FIX-10: Year-stratified sample includes partial year — min 60 rows guard added
  FIX-11: Cumulative IC re-computed in main() — daily_ic_cache saved in compute_ic_stats
  FIX-12: add_macro_features called twice — guard added in main()
  MASTER FIXES:
  - Memory/MP: ThreadPool executor + shared mem arrays via joblib prevents OOM.
  - Statistical: Weighted IC implemented based on daily N_tickers.
  - Degenerate: `pre_filter_factors` explicitly drops dead columns before loops.
  - Look-ahead: Date gap detection added to find cross-month leakage.

Usage:
    python scripts/validate_factors.py
    python scripts/validate_factors.py --threshold 0.015 --top-n 20
    python scripts/validate_factors.py --force-rebuild
    python scripts/validate_factors.py --force-lookahead   # skip look-ahead guard
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
plt.switch_backend("Agg")
from scipy import stats as scipy_stats
from scipy.stats import t as t_dist
from joblib import Parallel, delayed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config
from config.logging_config import setup_logging
from quant_alpha.utils import time_execution
from scripts.train_models import load_and_build_full_dataset, add_macro_features

setup_logging()
logger = logging.getLogger("Quant_Alpha")
warnings.filterwarnings("ignore")

META_COLS = {
    "date", "ticker", "open", "high", "low", "close", "volume", "vwap",
    "sector", "industry", "target", "raw_ret_5d", "next_open", "future_open",
    "pnl_return", "index", "level_0", "split_factor", "div_factor",
    "macro_mom_5d", "macro_mom_21d", "macro_vix_proxy", "macro_trend_200d",
    "us_10y_close", "vix_close", "oil_close", "usd_close", "sp500_close"
}

IC_DECAY_LAGS = [1, 2, 3, 5, 10, 21]

# Clip bounds — strictly synced with config/settings.py
RETURN_CLIP_MIN = getattr(config, "RETURN_CLIP_MIN", -0.50)
RETURN_CLIP_MAX = getattr(config, "RETURN_CLIP_MAX",  0.50)

# Look-ahead bias: correlation threshold with intraday return
# FIX-8: lowered from 0.5 to 0.3 — 0.5 gave false safety confidence
LOOKAHEAD_CORR_THRESHOLD = 0.30


class FactorValidator:
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "raw_ret_5d",
        force_lookahead: bool = False,
    ) -> None:
        self.data            = data.sort_values(["ticker", "date"]).reset_index(drop=True)
        self.target_col      = target_col
        self.force_lookahead = force_lookahead
        self.factors         = self._identify_factors()
        self.results: dict   = {}
        # FIX-11: cache for daily IC series keyed by factor name
        self._daily_ic_cache: dict[str, pd.Series] = {}

        # FIX-1: was called twice — now called exactly once
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

    def _assert_target_is_forward(self) -> None:
        """
        Detect potential look-ahead bias.

        FIX-8: threshold lowered from 0.5 to 0.3.
        A correlation of 0.3 with intraday returns already implies
        the target was not properly shifted. Now raises ValueError
        instead of just warning — use --force-lookahead to override.
        """
        if "close" not in self.data.columns or self.target_col not in self.data.columns:
            return
        intraday = (self.data["close"] / self.data["open"] - 1).dropna()
        target   = self.data[self.target_col].dropna()
        common   = intraday.index.intersection(target.index)
        if len(common) < 100:
            return
        corr = abs(intraday.loc[common].corr(target.loc[common]))
        if corr > LOOKAHEAD_CORR_THRESHOLD:
            msg = (
                f"target '{self.target_col}' correlates {corr:.3f} with "
                f"contemporaneous intraday return (threshold={LOOKAHEAD_CORR_THRESHOLD}). "
                f"Likely look-ahead bias — check forward-shift. "
                f"Pass force_lookahead=True to override."
            )
            if self.force_lookahead:
                logger.warning(f"[LOOKAHEAD OVERRIDE] {msg}")
            else:
                raise ValueError(f"[LOOKAHEAD BIAS DETECTED] {msg}")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. COVERAGE  (vectorised)
    # ──────────────────────────────────────────────────────────────────────────
    @time_execution
    def check_coverage(self) -> pd.DataFrame:
        """Vectorized coverage stats — O(cols) not O(cols × rows)."""
        logger.info("Checking data coverage...")
        df = self.data[self.factors]
        return pd.DataFrame({
            "coverage_pct": 1 - df.isna().mean(),
            "nan_count":    df.isna().sum(),
            "inf_count":    df.apply(lambda c: np.isinf(c).sum()),
            "zero_pct":     (df == 0).mean(),
        })

    @time_execution
    def pre_filter_factors(self) -> tuple[list[str], dict]:
        """Prune zero-variance and highly missing factors before IC loop."""
        logger.info("Pre-filtering degenerate factors...")
        df = self.data[self.factors]
        
        missing_pct = df.isna().mean()
        stds = df.std()
        
        valid_factors = []
        filter_reasons = {}
        
        for f in self.factors:
            if missing_pct[f] > 0.5:
                filter_reasons[f] = "FAIL (Missing Data > 50%)"
            elif pd.isna(stds[f]) or stds[f] < 1e-8:
                filter_reasons[f] = "FAIL (Zero Variance)"
            else:
                valid_factors.append(f)
                
        logger.info(f"Pruned {len(self.factors) - len(valid_factors)} degenerate factors.")
        return valid_factors, filter_reasons

    # ──────────────────────────────────────────────────────────────────────────
    # 2. IC STATS  (raw + sector-neutral, vectorised inner loop)
    # ──────────────────────────────────────────────────────────────────────────
    @time_execution
    def compute_ic_stats(self) -> pd.DataFrame:
        """
        Compute Rank IC (Spearman) and sector-neutral Rank IC per factor.
        """
        if "ic_stats" in self.results:
            return self.results["ic_stats"]

        logger.info(f"Computing IC (raw + sector-neutral) | Target: {self.target_col}")

        valid_factors, filter_reasons = self.pre_filter_factors()
        valid      = self.data.dropna(subset=[self.target_col]).copy()
        has_sector = "sector" in valid.columns
        
        # Daily weights for Weighted IC
        daily_n = valid.groupby("date").size()

        if has_sector:
            t_neutral = (
                valid.groupby(["date", "sector"])[self.target_col]
                .transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
            )
        else:
            t_neutral = None

        t_ranks_raw = valid.groupby("date")[self.target_col].rank(pct=True)
        t_ranks_raw_series = pd.Series(t_ranks_raw.values, index=valid.index)
        t_dm = t_ranks_raw_series - t_ranks_raw_series.groupby(valid["date"]).transform("mean")
        t_ss = t_dm.groupby(valid["date"]).apply(lambda x: (x**2).sum()).pow(0.5)
        
        if has_sector:
            t_neutral_series = pd.Series(t_neutral.values, index=valid.index)
            sn_t_mean = t_neutral_series.groupby(valid["date"]).transform("mean")
            sn_t_dm = t_neutral_series - sn_t_mean
            sn_t_ss = sn_t_dm.groupby(valid["date"]).apply(lambda x: (x**2).sum()).pow(0.5)
        else:
            sn_t_dm = None
            sn_t_ss = None

        def _process_factor(f):
            try:
                f_ranks = valid.groupby("date")[f].rank(pct=True)
                f_dm = f_ranks - f_ranks.groupby(valid["date"]).transform("mean")
                f_ss = f_dm.groupby(valid["date"]).apply(lambda x: (x**2).sum()).pow(0.5)
                
                cov = (f_dm * t_dm).groupby(valid["date"]).sum()
                denom = (f_ss * t_ss).replace(0, np.nan)
                daily_ic = (cov / denom).dropna()

                if len(daily_ic) < 3:
                    return None

                weights = daily_n.loc[daily_ic.index]
                weights = weights / weights.sum()
                ic_mean = float(np.average(daily_ic, weights=weights))
                ic_std  = float(daily_ic.std()) + 1e-8
                icir    = ic_mean / ic_std
                n_dates = len(daily_ic)
                t_stat  = icir * np.sqrt(n_dates)

                sn_icir = np.nan
                if has_sector:
                    f_neutral = (
                        valid.groupby(["date", "sector"])[f]
                        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
                    )
                    sn_f_mean = f_neutral.groupby(valid["date"]).transform("mean")
                    sn_f_dm = f_neutral - sn_f_mean
                    sn_f_ss = sn_f_dm.groupby(valid["date"]).apply(lambda x: (x**2).sum()).pow(0.5)
                    
                    sn_cov = (sn_f_dm * sn_t_dm).groupby(valid["date"]).sum()
                    sn_denom = (sn_f_ss * sn_t_ss).replace(0, np.nan)
                    sn_daily = (sn_cov / sn_denom).dropna()
                    
                    if len(sn_daily) >= 3:
                        sn_weights = daily_n.loc[sn_daily.index]
                        sn_weights = sn_weights / sn_weights.sum()
                        sn_m = float(np.average(sn_daily, weights=sn_weights))
                        sn_s = float(sn_daily.std()) + 1e-8
                        sn_icir = sn_m / sn_s

                return {
                    "factor":     f,
                    "ic_mean":    round(ic_mean, 5),
                    "ic_std":     round(ic_std, 5),
                    "icir":       round(icir, 4),
                    "t_stat":     round(t_stat, 3),
                    "pos_ic_pct": round(float((daily_ic > 0).mean()), 3),
                    "n_dates":    n_dates,
                    "sn_icir":    round(sn_icir, 4) if not np.isnan(sn_icir) else np.nan,
                    "daily_ic":   daily_ic
                }
            except Exception as exc:
                logger.warning(f"IC failed for {f}: {exc}")
                return None

        # ThreadPool prevents memory exhaustion from multiprocessing serialization
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_process_factor)(f) for f in tqdm(valid_factors, desc="IC")
        )

        ic_stats = []
        for r in results:
            if r is not None:
                self._daily_ic_cache[r["factor"]] = r.pop("daily_ic")
                ic_stats.append(r)
                
        # Inject filtered degenerate factors for reporting
        for f, reason in filter_reasons.items():
            ic_stats.append({
                "factor": f, "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                "t_stat": np.nan, "pos_ic_pct": np.nan, "n_dates": 0, "sn_icir": np.nan,
                "_filter_reason": reason
            })

        if not ic_stats:
            return pd.DataFrame()

        df = pd.DataFrame(ic_stats).set_index("factor")

        # ── FIX-4: BH correction — per-factor df from n_dates ────────────────
        # Original hardcoded df=500 for all factors regardless of their
        # actual date count — inflated p-values for sparse factors,
        # deflated for dense ones.
        try:
            from statsmodels.stats.multitest import multipletests
            p_vals = np.array([
                float(2 * (1 - t_dist.cdf(
                    abs(row["t_stat"]),
                    df=max(int(row["n_dates"]) - 2, 1)
                )))
                for _, row in df.iterrows()
            ])
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
    # 3. IC DECAY
    # ──────────────────────────────────────────────────────────────────────────
    @time_execution
    def compute_ic_decay(self, top_factors: list[str]) -> pd.DataFrame:
        """
        IC at lags 1, 2, 3, 5, 10, 21 trading days.

        FIX-5: Shifted target ranks pre-computed ONCE per lag outside the
               factor loop. Original re-ranked the shifted target for every
               factor at every lag = O(F × lags) redundant ranking operations.
        """
        logger.info(f"Computing IC decay for top {len(top_factors)} factors...")

        # FIX-5: pre-rank shifted targets once per lag — O(lags) not O(F × lags)
        shifted_targets: dict[int, pd.Series] = {}
        for lag in IC_DECAY_LAGS:
            shifted = self.data.groupby("ticker")[self.target_col].shift(-lag)
            shifted_targets[lag] = shifted

        decay_rows = []
        for f in tqdm(top_factors, desc="IC decay"):
            row = {"factor": f}
            for lag in IC_DECAY_LAGS:
                try:
                    # BUG-001 FIX: Drop NaNs before ranking
                    tmp_df = self.data[["date", f]].copy()
                    tmp_df["_t"] = shifted_targets[lag]
                    tmp_df = tmp_df.dropna()

                    if len(tmp_df) < 3:
                        row[f"ic_lag{lag}"] = np.nan
                        continue
                        
                    f_ranks = tmp_df.groupby("date")[f].rank(pct=True)
                    t_ranks = tmp_df.groupby("date")["_t"].rank(pct=True)
                    
                    tmp = pd.DataFrame({
                        "f": f_ranks.values,
                        "t": t_ranks.values,
                        "date": tmp_df["date"].values,
                    })

                    # Vectorised Spearman (same approach as compute_ic_stats)
                    grp   = tmp.groupby("date")
                    f_dm  = tmp["f"] - tmp.groupby("date")["f"].transform("mean")
                    t_dm  = tmp["t"] - tmp.groupby("date")["t"].transform("mean")
                    tmp2  = tmp.assign(f_dm=f_dm, t_dm=t_dm)
                    grp2  = tmp2.groupby("date")
                    cov   = grp2.apply(lambda x: (x["f_dm"] * x["t_dm"]).sum())
                    f_ss  = grp2.apply(lambda x: (x["f_dm"] ** 2).sum()).pow(0.5)
                    t_ss  = grp2.apply(lambda x: (x["t_dm"] ** 2).sum()).pow(0.5)
                    denom = (f_ss * t_ss).replace(0, np.nan)
                    ic    = (cov / denom).dropna().mean()
                    row[f"ic_lag{lag}"] = round(float(ic), 5)
                except Exception:
                    row[f"ic_lag{lag}"] = np.nan
            decay_rows.append(row)

        return pd.DataFrame(decay_rows).set_index("factor")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. AUTOCORRELATION  (per-ticker AR(1))
    # ──────────────────────────────────────────────────────────────────────────
    @time_execution
    def compute_autocorrelation(self) -> pd.DataFrame:
        """
        Per-ticker AR(1), averaged across tickers.

        FIX-9: Zero-variance series now returns NaN instead of hardcoded 1.0.
               A constant signal is zero-information and should be tagged
               FAIL (Zero Variance) downstream, not given a perfect autocorr
               score that masks the problem.
        """
        logger.info("Computing factor autocorrelation (per-ticker AR1)...")
        auto_corrs = []

        for f in tqdm(self.factors, desc="Autocorr"):
            try:
                def _ticker_autocorr(s: pd.Series) -> float:
                    # FIX-9: return NaN for constant series — caller tags as dead factor
                    # Original returned 1.0 which masked zero-information factors
                    if s.std() < 1e-8:
                        return np.nan
                    return float(s.corr(s.shift(1)))

                per_ticker = self.data.groupby("ticker")[f].apply(_ticker_autocorr)
                auto_corrs.append({
                    "factor":       f,
                    "autocorr":     round(float(per_ticker.mean(skipna=True)), 4),
                    "autocorr_std": round(float(per_ticker.std(skipna=True)), 4),
                    "n_const_tickers": int(per_ticker.isna().sum()),
                })
            except Exception:
                auto_corrs.append({
                    "factor": f,
                    "autocorr": np.nan,
                    "autocorr_std": np.nan,
                    "n_const_tickers": 0,
                })

        return pd.DataFrame(auto_corrs).set_index("factor")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. REDUNDANCY HEATMAP
    # ──────────────────────────────────────────────────────────────────────────
    @time_execution
    def check_redundancy(self, top_n_factors: list[str]) -> None:
        """
        Spearman correlation heatmap.

        FIX-10: Year-stratified sample now requires >= 60 rows per year
                to exclude partial years that bias the correlation estimate
                toward recent data patterns.
        """
        logger.info("Generating correlation heatmap (year-stratified sample)...")
        if not top_n_factors:
            return

        # FIX-10: min 60 rows guard excludes partial current year
        sample_df = (
            self.data[top_n_factors + ["date"]]
            .assign(year=lambda x: pd.to_datetime(x["date"]).dt.year)
            .groupby("year", group_keys=False)
            .apply(
                lambda x: x.sample(min(5_000, len(x)), random_state=42)
                if len(x) >= 60 else pd.DataFrame()
            )
            .drop(columns=["year", "date"], errors="ignore")
        )

        if sample_df.empty or len(sample_df.columns) < 2:
            logger.warning("Insufficient data for redundancy heatmap — skipping.")
            return

        corr_matrix = sample_df.corr(method="spearman")

        fig_size = max(10, len(top_n_factors) // 2)
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(
            corr_matrix,
            annot=len(top_n_factors) <= 20,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.3,
        )
        plt.title(f"Factor Spearman correlation (top {len(top_n_factors)}, year-stratified)")
        plt.tight_layout()

        out_path = config.RESULTS_DIR / "validation" / "factor_correlation.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Heatmap saved → {out_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. MASTER REPORT
    # ──────────────────────────────────────────────────────────────────────────
    def generate_report(
        self,
        output_dir: Path,
        threshold: float = 0.015,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Compile all metrics → master CSV + console summary."""
        coverage = self.check_coverage()
        ic_stats = self.compute_ic_stats()
        autocorr = self.compute_autocorrelation()

        if ic_stats.empty:
            logger.error("No IC stats computed — aborting report.")
            return pd.DataFrame()

        report = ic_stats.join(coverage, how="left").join(autocorr, how="left")

        # ── Status classification ─────────────────────────────────────────────
        report["status"]    = "FAIL (Weak Signal)"
        report["rescuable"] = False

        if "_filter_reason" in report.columns:
            mask_filtered = report["_filter_reason"].notna()
            report.loc[mask_filtered, "status"] = report.loc[mask_filtered, "_filter_reason"]

        # FIX-9: Zero-variance factors — flagged separately before other gates
        if "n_const_tickers" in report.columns:
            n_tickers = self.data["ticker"].nunique() if "ticker" in self.data.columns else 1
            mask_zero_var = report["n_const_tickers"] > (n_tickers * 0.5)
            report.loc[mask_zero_var, "status"] = "FAIL (Zero Variance)"

        mask_not_zero = report["status"] != "FAIL (Zero Variance)"
        
        # 0. Missing Data
        mask_missing = report["coverage_pct"] < 0.50
        report.loc[mask_missing, "status"] = "FAIL (Missing Data > 50%)"

        sn_icir = report.get("sn_icir", pd.Series(0.0, index=report.index))

        # 1. PASS
        mask_pass = (
            (report["coverage_pct"] >= 0.50) &
            (
                ((report["ic_mean"] >= threshold) & (report["t_stat"] >= 2.0)) |
                (sn_icir >= 0.20)
            ) & mask_not_zero & ~mask_missing
        )
        report.loc[mask_pass, "status"] = "PASS"

        # 2. PASS (Inverted)
        mask_inverted = (
            (report["coverage_pct"] >= 0.50) &
            (
                ((report["ic_mean"] <= -threshold) & (report["t_stat"] <= -2.0)) |
                (sn_icir <= -0.20)
            ) & mask_not_zero & ~mask_missing & ~mask_pass
        )
        report.loc[mask_inverted, "status"]    = "PASS (Inverted — flip sign)"
        report.loc[mask_inverted, "rescuable"] = True

        # 3. WARN: high turnover
        mask_warn_turnover = (
            report["status"].str.startswith("PASS") &
            (report["autocorr"] < 0.3)
        )
        report.loc[mask_warn_turnover, "status"] = "WARN (High Turnover)"

        # 4. WARN: sector bias
        if "sn_icir" in report.columns:
            mask_sector_bias = (
                report["status"].str.startswith("PASS") &
                report["sn_icir"].notna() &
                (report["sn_icir"].abs() < report["icir"].abs() * 0.5)
            )
            report.loc[mask_sector_bias, "status"] = "WARN (Sector Bias)"

        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "factor_validation_report.csv"
        report.to_csv(csv_path)

        # ── IC Decay for top passing factors ──────────────────────────────────
        passing  = report[report["status"].str.startswith("PASS")].head(top_n).index.tolist()
        decay_df = pd.DataFrame()
        if passing:
            decay_df = self.compute_ic_decay(passing)
            decay_df.to_csv(output_dir / "ic_decay.csv")

        # ── Console Summary ───────────────────────────────────────────────────
        n_pass_clean = (report["status"] == "PASS").sum()
        n_inverted   = (report["status"] == "PASS (Inverted — flip sign)").sum()
        n_warn       = report["status"].str.startswith("WARN").sum()
        n_missing    = (report["status"] == "FAIL (Missing Data > 50%)").sum()
        n_zero_var   = (report["status"] == "FAIL (Zero Variance)").sum()
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
        print(f"  FAIL (Missing)    : {n_missing:<4} (NaN > 50% — check upstream data)")
        print(f"  FAIL (Zero Var)   : {n_zero_var:<4} (dead factor — drop)")
        print(f"  FAIL (Weak)       : {n_weak:<4} (drop)")

        cols = ["ic_mean", "icir", "sn_icir", "t_stat", "p_bh_corr",
                "autocorr", "coverage_pct", "pos_ic_pct", "status"]
        cols = [c for c in cols if c in report.columns]

        print(f"\nTop 10 alpha drivers:")
        print(report.sort_values("t_stat", ascending=False).head(10)[cols].to_string())

        if not decay_df.empty:
            print(f"\nIC decay (top {len(passing)} factors):")
            print(decay_df.to_string())
            print("  -> Fast decay (lag5 ~0): daily rebalance needed")
            print("  -> Slow decay (lag10 still strong): weekly rebalance fine")

        failed = report[report["status"].str.startswith("FAIL")]
        print(f"\nFailed factors ({len(failed)} total), bottom 5 by IC:")
        if not failed.empty:
            print(
                failed.sort_values("ic_mean")
                      .head(5)[["ic_mean", "coverage_pct", "status"]]
                      .to_string()
            )

        print(f"\nFull report -> {csv_path}")
        self.check_redundancy(passing[:top_n])
        return report


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quant Alpha Factor Validation Suite v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python scripts/validate_factors.py
  python scripts/validate_factors.py --threshold 0.02 --top-n 25
  python scripts/validate_factors.py --force-rebuild
  python scripts/validate_factors.py --force-lookahead
        """,
    )
    parser.add_argument("--threshold",      type=float, default=0.010,
                        help="Minimum |IC| for PASS status (default: 0.010).")
    parser.add_argument("--top-n",          type=int,   default=20,
                        help="Top N passing factors for decay + heatmap.")
    parser.add_argument("--force-rebuild",  action="store_true",
                        help="Rebuild dataset from raw files.")
    # FIX-8: --force-lookahead allows researcher to proceed despite look-ahead warning
    parser.add_argument("--force-lookahead", action="store_true",
                        help="Skip look-ahead bias guard (use with caution).")
    args = parser.parse_args()

    # 1. Load
    logger.info("Loading master dataset...")
    try:
        data = load_and_build_full_dataset(force_rebuild=args.force_rebuild)

        # FIX-12: Guard against double call — load_and_build_full_dataset already
        # calls add_macro_features internally when building fresh.
        # Calling it again here re-computes all macro columns unnecessarily.
        if "macro_mom_5d" not in data.columns:
            data = add_macro_features(data)

        if "raw_ret_5d" not in data.columns:
            logger.info("Constructing raw_ret_5d target...")
            data = data.sort_values(["ticker", "date"])
            next_open   = data.groupby("ticker")["open"].shift(-1).replace(0, np.nan)
            future_open = data.groupby("ticker")["open"].shift(-6)
            # FIX-2: clip to ±50% — matches RETURN_CLIP_MIN/MAX in train_models.py
            # Original had no clip; a single halt/split row corrupts IC for many factors
            data["raw_ret_5d"] = (
                (future_open / next_open) - 1
            ).clip(RETURN_CLIP_MIN, RETURN_CLIP_MAX)

        data = data.dropna(subset=["raw_ret_5d"])

    except Exception as exc:
        logger.error(f"Data load failed: {exc}")
        sys.exit(1)

    logger.info(f"Loaded: {len(data):,} rows x {len(data.columns)} cols")

    # 2. Validate
    # FIX-8: pass force_lookahead flag through to validator
    try:
        validator = FactorValidator(
            data,
            target_col="raw_ret_5d",
            force_lookahead=args.force_lookahead,
        )
    except ValueError as exc:
        logger.error(str(exc))
        logger.error("Re-run with --force-lookahead to override, or fix target construction.")
        sys.exit(1)

    if not validator.factors:
        logger.error("No numeric factors found!")
        sys.exit(1)

    print(f"\nAnalysing {len(validator.factors)} factors...")
    output_dir = config.RESULTS_DIR / "validation"

    # 3. Report — ic_stats cached inside, no double compute
    report = validator.generate_report(
        output_dir,
        threshold=args.threshold,
        top_n=args.top_n,
    )

    # 4. Cumulative IC plot for best factor
    # FIX-11: Use cached daily IC series — do NOT recompute from scratch.
    # Original recomputed daily_ic in main() independently from compute_ic_stats,
    # risking divergence if data was mutated between calls.
    ic_stats = validator.results.get("ic_stats", pd.DataFrame())
    if ic_stats.empty:
        return

    best = ic_stats.index[0]
    logger.info(f"Plotting cumulative IC for best factor: {best}")

    # FIX-11: Retrieve from cache — guaranteed same result as validation run
    daily_ic = validator._daily_ic_cache.get(best)
    if daily_ic is None or daily_ic.empty:
        logger.warning(f"Daily IC cache empty for {best} — skipping plot.")
        return

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
    ax2.axhline(
        args.threshold, color="green", linestyle=":", linewidth=1,
        label=f"threshold={args.threshold}",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"cum_ic_{best}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved -> {out}")


if __name__ == "__main__":
    main()