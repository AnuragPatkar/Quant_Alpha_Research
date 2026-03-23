"""
Factor Quality Assurance and Validation Suite
=============================================
Evaluates the statistical efficacy and robustness of engineered alpha factors.

Purpose
-------
This module computes rigorous cross-sectional Information Coefficient (IC) 
metrics, temporal signal decay, and collinearity profiles for all candidate
features. It enforces strict guards against look-ahead bias and structural 
degeneracy to ensure only robust signals are passed to the machine learning
ensemble.

Role in Quantitative Workflow
-----------------------------
Executed post-feature engineering and prior to model training. Acts as the 
primary statistical gatekeeper, generating the `factor_validation_report.csv` 
used to define the active feature set for the predictive modeling layer.

Dependencies
------------
- **Pandas/NumPy**: Vectorized ranking and cross-sectional matrix operations.
- **SciPy/Statsmodels**: T-statistics and Benjamini-Hochberg FDR correction.
- **Joblib**: Parallel processing for computationally intensive IC calculations.
- **Matplotlib/Seaborn**: Heatmap and decay visualizations.
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

# Establishes structural clipping thresholds exactly mirrored from the master configuration
RETURN_CLIP_MIN = getattr(config, "RETURN_CLIP_MIN", -0.50)
RETURN_CLIP_MAX = getattr(config, "RETURN_CLIP_MAX",  0.50)

# Establishes a highly restrictive threshold against contemporaneous correlation leakage
LOOKAHEAD_CORR_THRESHOLD = 0.30


class FactorValidator:
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "raw_ret_5d",
        force_lookahead: bool = False,
    ) -> None:
        """
        Initializes the validation state matrix and evaluates structural prerequisites.

        Args:
            data (pd.DataFrame): Aggregated master dataset containing feature histories.
            target_col (str, optional): The independent variable targeted for prediction. Defaults to "raw_ret_5d".
            force_lookahead (bool, optional): Allows validation to proceed despite detected leakage. Defaults to False.
        """
        self.data            = data.sort_values(["ticker", "date"]).reset_index(drop=True)
        self.target_col      = target_col
        self.force_lookahead = force_lookahead
        self.factors         = self._identify_factors()
        self.results: dict   = {}
        
        # Caches daily IC series sequentially mapped by factor name to bypass redundant computation
        self._daily_ic_cache: dict[str, pd.Series] = {}

        # Triggers look-ahead bias detection during initialization
        self._assert_target_is_forward()

    def _identify_factors(self) -> list[str]:
        """
        Isolates strictly numerical candidate columns, bypassing metadata constraints.

        Args:
            None

        Returns:
            list[str]: Array of target features valid for statistical analysis.
        """
        return [
            c for c in self.data.columns
            if c not in META_COLS
            and not c.startswith("_")
            and pd.api.types.is_numeric_dtype(self.data[c])
        ]

    def _assert_target_is_forward(self) -> None:
        """
        Evaluates the dataset for potential look-ahead bias by correlating the target 
        with contemporaneous intraday returns.

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: If the absolute correlation exceeds the established threshold,
                indicating future data leakage into the target variable.
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

    @time_execution
    def check_coverage(self) -> pd.DataFrame:
        """
        Computes vectorized sparsity arrays evaluating null boundaries globally.

        Args:
            None

        Returns:
            pd.DataFrame: Metric distributions capturing structural feature voids.
        """
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
        """
        Evaluates structural matrices, pruning highly sparse or absolutely zero-variance 
        signals before committing to intensive calculation cycles.

        Args:
            None

        Returns:
            tuple[list[str], dict]: Extracted valid factor subsets alongside disqualification reasoning.
        """
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

    @time_execution
    def compute_ic_stats(self) -> pd.DataFrame:
        """
        Constructs absolute Rank IC (Spearman) boundaries dynamically capturing specific 
        cross-sectional efficacy parameters for both raw and sector-neutral distributions.

        Args:
            None

        Returns:
            pd.DataFrame: A comprehensive matrix of statistical outcomes dictating 
                signal directionality, significance, and relative breadth.
        """
        if "ic_stats" in self.results:
            return self.results["ic_stats"]

        logger.info(f"Computing IC (raw + sector-neutral) | Target: {self.target_col}")

        valid_factors, filter_reasons = self.pre_filter_factors()
        valid      = self.data.dropna(subset=[self.target_col]).copy()
        has_sector = "sector" in valid.columns
        
        # Daily volume vectors to facilitate precise breadth-weighted evaluations
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

        # Implements threaded map execution to mitigate IPC boundaries causing systemic OOMs
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_process_factor)(f) for f in tqdm(valid_factors, desc="IC")
        )

        ic_stats = []
        for r in results:
            if r is not None:
                self._daily_ic_cache[r["factor"]] = r.pop("daily_ic")
                ic_stats.append(r)
                
        # Injects disqualified factor configurations universally to maintain reporting dimensions
        for f, reason in filter_reasons.items():
            ic_stats.append({
                "factor": f, "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                "t_stat": np.nan, "pos_ic_pct": np.nan, "n_dates": 0, "sn_icir": np.nan,
                "_filter_reason": reason
            })

        if not ic_stats:
            return pd.DataFrame()

        df = pd.DataFrame(ic_stats).set_index("factor")

        # Applies Benjamini-Hochberg False Discovery Rate (FDR) correction utilizing dynamic degrees of freedom
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

    @time_execution
    def compute_ic_decay(self, top_factors: list[str]) -> pd.DataFrame:
        """
        Computes Information Coefficient measurements across specific target lags 
        to identify the half-life and alpha turnover boundaries for prime candidates.

        Args:
            top_factors (list[str]): The subset array of top candidate signals to examine.

        Returns:
            pd.DataFrame: Temporally scaled coefficient observations mapped across specified delays.
        """
        logger.info(f"Computing IC decay for top {len(top_factors)} factors...")

        # Pre-computes shifted target ranks universally across lags to optimize temporal complexity
        shifted_targets: dict[int, pd.Series] = {}
        for lag in IC_DECAY_LAGS:
            shifted = self.data.groupby("ticker")[self.target_col].shift(-lag)
            shifted_targets[lag] = shifted

        decay_rows = []
        for f in tqdm(top_factors, desc="IC decay"):
            row = {"factor": f}
            for lag in IC_DECAY_LAGS:
                try:
                    # Strictly excludes non-finite observations prior to cross-sectional ranking to prevent distributional skew
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

                    # Derives Spearman via vectorized distribution parameters
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

    @time_execution
    def compute_autocorrelation(self) -> pd.DataFrame:
        """
        Quantifies the AR(1) signal stability parameters universally across execution arrays.

        Args:
            None

        Returns:
            pd.DataFrame: A matrix of calculated day-over-day turnover estimations indicating execution cost bounds.
        """
        logger.info("Computing factor autocorrelation (per-ticker AR1)...")
        auto_corrs = []

        for f in tqdm(self.factors, desc="Autocorr"):
            try:
                def _ticker_autocorr(s: pd.Series) -> float:
                    # Evaluates and assigns NaN to zero-variance series to explicitly flag zero-information signals
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

    @time_execution
    def check_redundancy(self, top_n_factors: list[str]) -> None:
        """
        Computes macro-stratified correlation measurements against target factors 
        to proactively map ensemble collinearity risks.

        Args:
            top_n_factors (list[str]): Highly-scoring candidate series to evaluate.

        Returns:
            None: Renders generated figures directly into the platform output path.
        """
        logger.info("Generating correlation heatmap (year-stratified sample)...")
        if not top_n_factors:
            return

        # Enforces a strict observation floor to prevent partial-year structural biases in correlation estimations
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

    def generate_report(
        self,
        output_dir: Path,
        threshold: float = 0.015,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Aggregates disparate testing arrays into unified management and execution summaries.

        Args:
            output_dir (Path): Destination registry where the generated artifacts map.
            threshold (float, optional): IC configuration mapping strict thresholds. Defaults to 0.015.
            top_n (int, optional): Evaluation constraint mapping decay sequences. Defaults to 20.

        Returns:
            pd.DataFrame: Absolute comprehensive matrix indicating the status parameters universally.
        """
        coverage = self.check_coverage()
        ic_stats = self.compute_ic_stats()
        autocorr = self.compute_autocorrelation()

        if ic_stats.empty:
            logger.error("No IC stats computed — aborting report.")
            return pd.DataFrame()

        report = ic_stats.join(coverage, how="left").join(autocorr, how="left")

        report["status"]    = "FAIL (Weak Signal)"
        report["rescuable"] = False

        if "_filter_reason" in report.columns:
            mask_filtered = report["_filter_reason"].notna()
            report.loc[mask_filtered, "status"] = report.loc[mask_filtered, "_filter_reason"]

        # Isolates and explicitly tags absolute zero-variance factors prior to applying general performance gates
        if "n_const_tickers" in report.columns:
            n_tickers = self.data["ticker"].nunique() if "ticker" in self.data.columns else 1
            mask_zero_var = report["n_const_tickers"] > (n_tickers * 0.5)
            report.loc[mask_zero_var, "status"] = "FAIL (Zero Variance)"

        mask_not_zero = report["status"] != "FAIL (Zero Variance)"
        
        mask_missing = report["coverage_pct"] < 0.50
        report.loc[mask_missing, "status"] = "FAIL (Missing Data > 50%)"

        sn_icir = report.get("sn_icir", pd.Series(0.0, index=report.index))

        mask_pass = (
            (report["coverage_pct"] >= 0.50) &
            (
                ((report["ic_mean"] >= threshold) & (report["t_stat"] >= 2.0)) |
                (sn_icir >= 0.20)
            ) & mask_not_zero & ~mask_missing
        )
        report.loc[mask_pass, "status"] = "PASS"

        mask_inverted = (
            (report["coverage_pct"] >= 0.50) &
            (
                ((report["ic_mean"] <= -threshold) & (report["t_stat"] <= -2.0)) |
                (sn_icir <= -0.20)
            ) & mask_not_zero & ~mask_missing & ~mask_pass
        )
        report.loc[mask_inverted, "status"]    = "PASS (Inverted — flip sign)"
        report.loc[mask_inverted, "rescuable"] = True

        mask_warn_turnover = (
            report["status"].str.startswith("PASS") &
            (report["autocorr"] < 0.3)
        )
        report.loc[mask_warn_turnover, "status"] = "WARN (High Turnover)"

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

        passing  = report[report["status"].str.startswith("PASS")].head(top_n).index.tolist()
        decay_df = pd.DataFrame()
        if passing:
            decay_df = self.compute_ic_decay(passing)
            decay_df.to_csv(output_dir / "ic_decay.csv")

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


def main() -> None:
    """
    Orchestrates the entire cross-sectional factor validation DAG.

    Args:
        None

    Returns:
        None
    """
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
    parser.add_argument("--force-lookahead", action="store_true",
                        help="Skip look-ahead bias guard (use with caution).")
    args = parser.parse_args()

    logger.info("Loading master dataset...")
    try:
        data = load_and_build_full_dataset(force_rebuild=args.force_rebuild)

        # Guards against redundant macro feature initialization when hydrating from cold starts
        if "macro_mom_5d" not in data.columns:
            data = add_macro_features(data)

        if "raw_ret_5d" not in data.columns:
            logger.info("Constructing raw_ret_5d target...")
            data = data.sort_values(["ticker", "date"])
            next_open   = data.groupby("ticker")["open"].shift(-1).replace(0, np.nan)
            future_open = data.groupby("ticker")["open"].shift(-6)
            # Restricts forward returns to defined thresholds to prevent anomaly propagation (e.g., structural gaps)
            data["raw_ret_5d"] = (
                (future_open / next_open) - 1
            ).clip(RETURN_CLIP_MIN, RETURN_CLIP_MAX)

        data = data.dropna(subset=["raw_ret_5d"])

    except Exception as exc:
        logger.error(f"Data load failed: {exc}")
        sys.exit(1)

    logger.info(f"Loaded: {len(data):,} rows x {len(data.columns)} cols")

    # Instantiates the validation suite, passing the structural look-ahead guard override if configured
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

    report = validator.generate_report(
        output_dir,
        threshold=args.threshold,
        top_n=args.top_n,
    )

    ic_stats = validator.results.get("ic_stats", pd.DataFrame())
    if ic_stats.empty:
        return

    best = ic_stats.index[0]
    logger.info(f"Plotting cumulative IC for best factor: {best}")

    # Retrieves specific daily IC series directly from the pre-computed state to prevent evaluation divergence
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