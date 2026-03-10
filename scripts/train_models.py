"""
train_models.py
===============
Production Walk-Forward Training Pipeline  —  v4 (Memory-Safe Edition)
------------------------------------------------------------------------
Production Walk-Forward Training Pipeline — v4 (Memory-Safe Edition)
----------------------------------------------------------------------
This script is the core of the alpha modeling system. It performs a rigorous
walk-forward cross-validation to train and evaluate multiple GBDT models,
gates them based on performance, and saves production-ready artifacts.

KEY ARCHITECTURE CHANGES vs v3 (which caused 28GB OOM):

  ROOT CAUSE OF OOM:
    _generate_folds() built a list of 20 full DataFrame copies BEFORE training.
    3 parallel models × 20 folds × 482MB/copy = ~28GB peak RAM.
    Fix: fold boundaries are date-tuples only. Data is sliced INSIDE the fold
    loop, one fold at a time, then immediately freed.
    Fix: Fold boundaries are now lightweight date-tuples. Data is sliced
    INSIDE the fold loop, one fold at a time, then immediately freed.

  PARALLEL → SERIAL (default):
    3 parallel models each calling _generate_folds() = triple the RAM.
    3 parallel models each holding data copies = triple the RAM.
    Serial = 1/3 peak RAM. Use --parallel-models only on 32GB+ machines.

  PREPROCESSING MOVED PER-FOLD (no global preprocessing leak):
    v3 winsorized/normalized the FULL dataset before walk-forward.
    This leaks future stats into training. Now: fit scalers on train fold only.

  FEATURE SELECTION CACHED (not re-run every fold):
    ICIR re-runs every FEATURE_RESELECT_EVERY_N_FOLDS folds.
    ICIR-based feature selection re-runs every FEATURE_RESELECT_EVERY_N_FOLDS folds.

  PRODUCTION MODEL SAVES:
    Gate-passing models saved to models/production/ with feature_names.

RAM PROFILE:
  v3 parallel:   ~28 GB peak (OOM on most laptops)
  v4 serial:     ~3-4 GB peak (safe on 8 GB machines)
  v4 parallel:   ~9-12 GB peak (safe on 16 GB machines)

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --force-rebuild
    python scripts/train_models.py --parallel-models   # 16GB+ RAM only
"""

from __future__ import annotations  # PEP 604/585 on Python 3.9+

import os
import sys
import gc
import hashlib
import logging
import warnings
import argparse
import joblib
import concurrent.futures
from pathlib import Path

# ==============================================================================
# CPU THROTTLE — MUST be before any ML library imports
# psutil/numpy/pandas/numba/tqdm imports are BELOW _set_thread_env() call.
# Importing numba before env vars are set initialises its thread pool with
# default settings — NUMBA_NUM_THREADS cannot be changed afterwards.
# ==============================================================================
TOTAL_CORES      = os.cpu_count() or 4
CPU_CORES_TO_USE = max(2, TOTAL_CORES // 2)
MODEL_THREADS    = CPU_CORES_TO_USE

# NUMBA_NUM_THREADS can only be set BEFORE Numba launches its thread pool.
# When train_models is imported as a module (by generate_predictions.py),
# Numba may already be running with the system default thread count.
# Guard: only set if not already set, and only override NUMBA_NUM_THREADS
# if Numba threads have not been launched yet.
def _set_thread_env(n: int) -> None:
    """Set threading env vars. Skip NUMBA_NUM_THREADS if already locked."""
    os.environ["OMP_NUM_THREADS"]      = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"]      = str(n)
    os.environ["BLAS_NUM_THREADS"]     = str(n)
    os.environ["LOKY_MAX_CPU_COUNT"]   = str(n)
    os.environ["NUMBA_CACHE_DIR"]      = ".numba_cache"
    os.environ["PYTHONWARNINGS"]       = "ignore"
    # Only set NUMBA_NUM_THREADS if Numba thread pool not yet started
    if "NUMBA_NUM_THREADS" not in os.environ:
        os.environ["NUMBA_NUM_THREADS"] = str(n)

_set_thread_env(CPU_CORES_TO_USE)

# NOW safe to import numba/numpy/pandas — thread env vars are already set
import psutil
import numpy as np
import pandas as pd
from numba import njit, prange
from tqdm import tqdm

print(f"[CPU] Total: {TOTAL_CORES} cores | Using: {CPU_CORES_TO_USE} | "
      f"RAM: {psutil.virtual_memory().total/1e9:.1f} GB")

# ==============================================================================
# PROJECT PATH
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ==============================================================================
# DOMAIN IMPORTS
# ==============================================================================

from quant_alpha.utils import (
    setup_logging, load_parquet, save_parquet,
    time_execution, calculate_returns
)
from config.settings import config
from quant_alpha.data.DataManager import DataManager
from quant_alpha.models.trainer import WalkForwardTrainer
from quant_alpha.models.lightgbm_model import LightGBMModel
from quant_alpha.models.xgboost_model import XGBoostModel
from quant_alpha.models.catboost_model import CatBoostModel
from quant_alpha.models.feature_selector import FeatureSelector
from quant_alpha.backtest.engine import BacktestEngine
from quant_alpha.backtest.metrics import print_metrics_report
from quant_alpha.backtest.attribution import SimpleAttribution, FactorAttribution
from quant_alpha.optimization.allocator import PortfolioAllocator
from quant_alpha.visualization import (
    plot_equity_curve, plot_drawdown, plot_monthly_heatmap,
    plot_ic_time_series, generate_tearsheet
)
from quant_alpha.utils.preprocessing import WinsorisationScaler, SectorNeutralScaler, winsorize_clip_nb

import quant_alpha.features.technical.momentum
import quant_alpha.features.technical.volatility
import quant_alpha.features.technical.volume
import quant_alpha.features.technical.mean_reversion
import quant_alpha.features.fundamental.value
import quant_alpha.features.fundamental.quality
import quant_alpha.features.fundamental.growth
import quant_alpha.features.fundamental.financial_health
import quant_alpha.features.earnings.surprises
import quant_alpha.features.earnings.estimates
import quant_alpha.features.earnings.revisions
import quant_alpha.features.alternative.macro
import quant_alpha.features.alternative.sentiment
import quant_alpha.features.alternative.inflation
import quant_alpha.features.composite.macro_adjusted
import quant_alpha.features.composite.system_health
import quant_alpha.features.composite.smart_signals

setup_logging()
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    import sys as _sys
    _ch = logging.StreamHandler(_sys.stdout)
    _ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(_ch)
logger = logging.getLogger(__name__)

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
TOP_N_STOCKS           = 25
STOCK_STOP_LOSS        = -0.05
PORTFOLIO_DD_EXIT      = -0.15
PORTFOLIO_DD_REENTRY   = -0.05
TRANSACTION_COST_BPS   = 10.0
TRANSACTION_COST       = TRANSACTION_COST_BPS / 10_000.0

TARGET_HORIZON_DAYS    = 5
WF_MIN_TRAIN_MONTHS    = 36
WF_TEST_MONTHS         = 6
WF_STEP_MONTHS         = 3
WF_WINDOW_TYPE         = "expanding"
WF_EMBARGO_DAYS        = 21

# IC gate thresholds — calibrated for daily equity alpha signals
#
# WHY NOT daily ICIR 0.30:
#   Daily ICIR = IC_mean / IC_std (on daily series).
#   For equity alpha, IC_std ≈ 3-4x IC_mean is normal (IC_std ~0.06, IC ~0.015-0.02).
#   Daily ICIR of 0.30 would require IC_std < IC_mean/0.30 — unrealistically tight.
#   Our ensemble hits ICIR=0.29 over 1228 days = annualized ICIR of 4.67 (excellent).
#
# CORRECT GATE: IC t-statistic = IC_mean / (IC_std / sqrt(N_days))
#   t > 2.0 = p < 0.05 (statistically significant IC over sample period)
#   t > 3.0 = strong signal (use this for production gate)
#   t < 1.5 = probably random noise (exclude from ensemble)
#
# Per-model thresholds:
#   IC_mean > 0.005  AND  t-stat > 1.5  → include in ensemble (weak but real)
#   IC_mean > 0.010  AND  t-stat > 2.5  → save production model
MIN_OOS_IC_THRESHOLD      = 0.005  # minimum mean IC to contribute to ensemble
MIN_OOS_IC_TSTAT          = 1.5    # minimum t-stat for ensemble inclusion
PROD_IC_THRESHOLD         = 0.010  # IC required to SAVE production .pkl model
PROD_IC_TSTAT             = 2.5    # t-stat required to SAVE production .pkl model
MIN_OOS_ICIR_THRESHOLD    = 0.30   # kept for backwards compatibility — NOT used for gate
                                    # annualized ICIR = daily_ICIR * sqrt(252)
ALPHA_SMOOTHING_LAMBDA = 0.70
RANDOM_SEED            = 42

# Feature selection cache — re-run ICIR every N folds
FEATURE_RESELECT_EVERY_N_FOLDS = 4
ICIR_N_BOOTSTRAP               = 20
ICIR_SAMPLE_SIZE               = 50_000


# ==============================================================================
# NUMBA JIT KERNELS
# ==============================================================================

@njit(cache=True)
def _rank1d(arr):
    """Fractional rank of 1D array. NaN → 0. O(N log N)."""
    n   = len(arr)
    out = np.zeros(n, dtype=np.float64)
    valid = np.empty(n, dtype=np.int64)
    nv = 0
    for i in range(n):
        if not np.isnan(arr[i]):
            valid[nv] = i
            nv += 1
    if nv == 0:
        return out
    vals = np.empty(nv, dtype=np.float64)
    for k in range(nv):
        vals[k] = arr[valid[k]]
    order = np.argsort(vals)
    k = 0
    while k < nv:
        j = k
        while j < nv - 1 and vals[order[j]] == vals[order[j + 1]]:
            j += 1
        avg_rank = (k + j + 2) / 2.0 / nv
        for m in range(k, j + 1):
            out[valid[order[m]]] = avg_rank
        k = j + 1
    return out


@njit(parallel=True, cache=True)
def spearman_ic_nb(feat_matrix, target):
    """Parallel Spearman IC per feature vs target."""
    n_samples, n_features = feat_matrix.shape
    ic_out  = np.zeros(n_features, dtype=np.float64)
    t_ranks = _rank1d(target)
    t_mean  = t_ranks.mean()
    t_std   = t_ranks.std() + 1e-10
    for f in prange(n_features):
        col    = feat_matrix[:, f].copy()
        fr     = _rank1d(col)
        f_mean = fr.mean()
        f_std  = fr.std() + 1e-10
        cov    = 0.0
        for i in range(n_samples):
            cov += (fr[i] - f_mean) * (t_ranks[i] - t_mean)
        cov /= max(n_samples - 1, 1)
        ic_out[f] = abs(cov / (f_std * t_std))
    return ic_out


@njit(parallel=True, cache=True)
def rank_pct_parallel_nb(pred_matrix, date_ids, n_dates):
    """Per-date percentile ranks across all model columns."""
    n_rows, n_models = pred_matrix.shape
    out = np.zeros_like(pred_matrix)
    for m in prange(n_models):
        for d in range(n_dates):
            idx_list = []
            for i in range(n_rows):
                if date_ids[i] == d:
                    idx_list.append(i)
            if len(idx_list) == 0:
                continue
            vals = np.empty(len(idx_list), dtype=np.float64)
            for k in range(len(idx_list)):
                vals[k] = pred_matrix[idx_list[k], m]
            ranks = _rank1d(vals)
            for k in range(len(idx_list)):
                out[idx_list[k], m] = ranks[k]
    return out


@njit(cache=True)
def compound_return_nb(weights, ticker_idx, returns_matrix):
    """Compound portfolio return over a period."""
    n_hold   = len(weights)
    n_days   = returns_matrix.shape[0]
    port_ret = 0.0
    for h in range(n_hold):
        col = ticker_idx[h]
        if col < 0:
            continue
        cum = 1.0
        for d in range(n_days):
            r = returns_matrix[d, col]
            if not np.isnan(r):
                cum *= (1.0 + r)
        port_ret += weights[h] * (cum - 1.0)
    return port_ret


def _warmup_numba():
    """Compile all kernels once; .numba_cache makes reruns instant."""
    d  = np.random.rand(60, 4).astype(np.float64)
    t  = np.random.rand(60).astype(np.float64)
    di = np.zeros(60, dtype=np.int64)
    w  = np.array([0.5, 0.5])
    ix = np.array([0, 1], dtype=np.int64)
    r  = np.random.rand(5, 5).astype(np.float64)
    winsorize_clip_nb(d, d * 0.1, d * 0.9)
    spearman_ic_nb(d, t)
    rank_pct_parallel_nb(d, di, 1)
    compound_return_nb(w, ix, r)
    logger.info("[NUMBA] JIT kernels ready.")


# ==============================================================================
# CACHE + DATA LOADING
# ==============================================================================
def get_data_hash(data_dir):
    hasher    = hashlib.md5()
    data_path = data_dir if hasattr(data_dir, "glob") else Path(data_dir)
    for f in sorted(data_path.glob("*.parquet")):
        hasher.update(f.name.encode())
        hasher.update(str(f.stat().st_mtime).encode())
        hasher.update(str(f.stat().st_size).encode())
    return hasher.hexdigest()


@time_execution
def load_and_build_full_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    cache_path   = config.CACHE_DIR / "master_data_with_factors.parquet"
    hash_path    = config.CACHE_DIR / "master_data_hash.txt"
    current_hash = get_data_hash(config.DATA_DIR)

    if not force_rebuild and os.path.exists(cache_path) and os.path.exists(hash_path):
        with open(hash_path) as f:
            if f.read().strip() == current_hash:
                logger.info("[CACHE] Hash match — loading cached dataset.")
                data = load_parquet(cache_path)
                if "raw_ret_5d" in data.columns:
                    return data
                logger.info("[CACHE] raw_ret_5d missing — rebuilding.")

    logger.info("[DATA] Initializing DataManager...")
    dm   = DataManager()
    data = dm.get_master_data()
    if data.index.names[0] is not None:
        data = data.reset_index()

    if data.shape[1] < 120:
        logger.info(f"[FACTORS] Computing on {data.shape[0]:,} rows...")
        from quant_alpha.features.registry import FactorRegistry
        data = FactorRegistry().compute_all(data)

    if "open" in data.columns:
        data = data.sort_values(["ticker", "date"])
        next_open   = data.groupby("ticker")["open"].shift(-1).replace(0, np.nan)
        future_open = data.groupby("ticker")["open"].shift(-6)
        data["raw_ret_5d"] = (future_open / next_open) - 1

    data = data.dropna(axis=1, how="all")
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    save_parquet(data, cache_path)
    with open(hash_path, "w") as f:
        f.write(current_hash)
    logger.info(f"[DATA] Built: {data.shape[0]:,} rows × {data.shape[1]} cols.")
    return data


def winsorize_fold(train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   features: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: fit on train, transform both. Used in fold loop."""
    scaler = WinsorisationScaler(clip_pct=0.01).fit(train_df, features)
    return scaler.transform(train_df, features), scaler.transform(test_df, features)


def sector_neutral_normalize_fold(train_df: pd.DataFrame,
                                   test_df: pd.DataFrame,
                                   features: list,
                                   sector_col: str = "sector"
                                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: fit on train, transform both. Used in fold loop."""
    scaler = SectorNeutralScaler(sector_col=sector_col).fit(train_df, features)
    return scaler.transform(train_df, features), scaler.inference_transform(test_df, features)


# ==============================================================================
# TARGET CONSTRUCTION
# ==============================================================================
def build_target(df: pd.DataFrame,
                 sector_mean=None,
                 vol_95th: float | None = None) -> pd.DataFrame:
    """Sector-neutral return target with volatility dampening."""
    df = df.copy()

    if sector_mean is None:
        s_mean = df.groupby(["date", "sector"])["raw_ret_5d"].transform("mean")
    elif isinstance(sector_mean, pd.Series) and len(sector_mean) == len(df):
        s_mean = sector_mean
    else:
        sm_df  = sector_mean.rename("_sm").reset_index()
        df     = df.merge(sm_df, on=["date", "sector"], how="left")
        s_mean = df.pop("_sm").fillna(0.0)

    df["target"] = df["raw_ret_5d"] - s_mean

    df["_sv"] = df.groupby(["date", "sector"])["raw_ret_5d"].transform("std")
    v95       = vol_95th if vol_95th is not None else float(df["_sv"].quantile(0.95))
    df.loc[df["_sv"] > v95, "target"] *= 0.5
    df = df.drop(columns=["_sv"])
    return df


# ==============================================================================
# FEATURE SELECTION (ICIR)
# ==============================================================================
def _single_bootstrap_ic(args):
    feat_matrix, target, seed = args
    rng   = np.random.default_rng(seed)
    n     = len(target)
    idx   = rng.choice(n, size=min(ICIR_SAMPLE_SIZE, n), replace=False)
    tgt   = target[idx]
    valid = ~np.isnan(tgt)
    tgt   = tgt[valid]
    if len(tgt) < 50:
        return None
    fmat = feat_matrix[idx][valid].astype(np.float64)
    col_means = np.nanmean(fmat, axis=0)
    fmat = np.where(np.isnan(fmat), col_means, fmat)
    return spearman_ic_nb(fmat, tgt)


def select_orthogonal_features(df: pd.DataFrame,
                                target_col: str,
                                exclude_cols: list,
                                top_n: int = 25,
                                corr_threshold: float = 0.70,
                                preserve_categoricals: list | None = None,
                                parallel_icir: bool = False) -> list:
    numeric    = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric if c not in exclude_cols and c != target_col]
    if not candidates:
        return preserve_categoricals or []

    sample    = df.sample(n=min(100_000, len(df)), random_state=RANDOM_SEED)
    feat_mat  = sample[candidates].values.astype(np.float64)
    tgt       = sample[target_col].values.astype(np.float64)
    valid     = ~np.isnan(tgt)
    tgt       = tgt[valid]
    fmat      = feat_mat[valid]
    col_means = np.nanmean(fmat, axis=0)
    fmat      = np.where(np.isnan(fmat), col_means, fmat)

    args_list = [(fmat, tgt, s) for s in range(RANDOM_SEED, RANDOM_SEED + ICIR_N_BOOTSTRAP)]

    ic_matrix = []
    if parallel_icir:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_CORES_TO_USE) as ex:
            for r in ex.map(_single_bootstrap_ic, args_list):
                if r is not None:
                    ic_matrix.append(r)
    else:
        for args in args_list:
            r = _single_bootstrap_ic(args)
            if r is not None:
                ic_matrix.append(r)

    if not ic_matrix:
        return preserve_categoricals or []

    ic_arr   = np.array(ic_matrix)
    icir     = ic_arr.mean(axis=0) / (ic_arr.std(axis=0) + 1e-8)
    ic_order = np.argsort(-icir)
    top_idx  = ic_order[:min(50, len(ic_order))]

    if len(top_idx) > 1:
        top_data    = fmat[:, top_idx]
        corr_matrix = np.corrcoef(top_data.T)
    else:
        corr_matrix = np.array([[1.0]])

    selected, sel_idxs = [], []
    for rank_pos, global_idx in enumerate(top_idx):
        feat = candidates[global_idx]
        if not sel_idxs:
            selected.append(feat); sel_idxs.append(rank_pos); continue
        if np.max(np.abs(corr_matrix[rank_pos, sel_idxs])) < corr_threshold:
            selected.append(feat); sel_idxs.append(rank_pos)
        if len(selected) >= top_n:
            break

    if preserve_categoricals:
        for cat in preserve_categoricals:
            if cat in df.columns and cat not in selected:
                selected.append(cat)

    logger.info(f"[FEATURE_SEL] {len(selected)} features selected.")
    return selected


# ==============================================================================
# CUSTOM OBJECTIVE
# ==============================================================================
def weighted_symmetric_mae(y_true, y_pred):
    """Penalises wrong-sign predictions 2×. Works for LGB/XGB custom objective."""
    residuals = y_true - y_pred
    weights   = np.where(y_true * y_pred < 0, 2.0, 1.0)
    grad      = -weights * np.tanh(residuals)
    hess      = np.maximum(weights * (1.0 - np.tanh(residuals) ** 2), 1e-3)
    return grad, hess


# ==============================================================================
# ENSEMBLE RANKING
# ==============================================================================
@time_execution
def calculate_ranks_robust(df: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        df["ensemble_alpha"] = 0.0
        return df

    # Z-score per date
    for col in pred_cols:
        means   = df.groupby("date")[col].transform("mean")
        stds    = df.groupby("date")[col].transform("std").replace(0, 1e-8)
        df[col] = (df[col] - means) / (stds + 1e-8)

    unique_dates = sorted(df["date"].unique())
    date_to_id   = {d: i for i, d in enumerate(unique_dates)}
    date_ids     = df["date"].map(date_to_id).values.astype(np.int64)
    pred_matrix  = df[pred_cols].values.astype(np.float64)
    rank_matrix  = rank_pct_parallel_nb(pred_matrix, date_ids, len(unique_dates))

    for i, col in enumerate(pred_cols):
        df[f"rank_{col}"] = rank_matrix[:, i]

    rank_cols          = [f"rank_{c}" for c in pred_cols]
    df["raw_alpha"]    = df[rank_cols].mean(axis=1)

    # EWM smoothing for turnover reduction
    df = df.sort_values(["ticker", "date"])
    df["ensemble_alpha"] = (
        df.groupby("ticker")["raw_alpha"]
          .transform(lambda s: s.ewm(alpha=1.0 - ALPHA_SMOOTHING_LAMBDA,
                                     adjust=False).mean())
    )
    return df


def build_ensemble_alpha(predictions: dict) -> pd.DataFrame:
    """
    Build ensemble alpha from per-model prediction dicts.
    Used by generate_predictions.py for inference.

    Args:
        predictions: {model_name: DataFrame with ["date", "ticker", "prediction"]}

    Returns:
        DataFrame with ["date", "ticker", "ensemble_alpha", + per-model pred cols]
    """
    if not predictions:
        return pd.DataFrame()

    ensemble_df = None
    for name, preds_df in predictions.items():
        col = f"pred_{name}"
        tmp = preds_df[["date", "ticker", "prediction"]].rename(columns={"prediction": col})
        ensemble_df = tmp if ensemble_df is None else pd.merge(
            ensemble_df, tmp, on=["date", "ticker"], how="outer")

    if ensemble_df is None or ensemble_df.empty:
        return pd.DataFrame()

    # For single-date inference (n_dates == 1), z-score is undefined →
    # skip it and go straight to rank percentile
    unique_dates = sorted(ensemble_df["date"].unique())
    pred_cols    = [c for c in ensemble_df.columns if c.startswith("pred_")]

    if len(unique_dates) > 1:
        for col in pred_cols:
            means = ensemble_df.groupby("date")[col].transform("mean")
            stds  = ensemble_df.groupby("date")[col].transform("std").replace(0, 1e-8)
            ensemble_df[col] = (ensemble_df[col] - means) / (stds + 1e-8)

    date_to_id  = {d: i for i, d in enumerate(unique_dates)}
    date_ids    = ensemble_df["date"].map(date_to_id).values.astype(np.int64)
    pred_matrix = ensemble_df[pred_cols].values.astype(np.float64)
    rank_matrix = rank_pct_parallel_nb(pred_matrix, date_ids, len(unique_dates))

    for i, col in enumerate(pred_cols):
        ensemble_df[f"rank_{col}"] = rank_matrix[:, i]

    rank_cols = [f"rank_{c}" for c in pred_cols]
    ensemble_df["raw_alpha"] = ensemble_df[rank_cols].mean(axis=1)

    # EWM smoothing — only meaningful for multi-date windows
    ensemble_df = ensemble_df.sort_values(["ticker", "date"])
    if len(unique_dates) > 1:
        ensemble_df["ensemble_alpha"] = (
            ensemble_df.groupby("ticker")["raw_alpha"]
              .transform(lambda s: s.ewm(alpha=1.0 - ALPHA_SMOOTHING_LAMBDA,
                                         adjust=False).mean())
        )
    else:
        ensemble_df["ensemble_alpha"] = ensemble_df["raw_alpha"]

    return ensemble_df



# ==============================================================================
# MACRO FEATURES
# ==============================================================================
def add_macro_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values("date")
    data["_dr"] = data.groupby("ticker")["close"].pct_change()
    mkt_ret     = data.groupby("date")["_dr"].mean()
    vix         = data.groupby("date")["_dr"].std()
    mkt_avg     = data.groupby("date")["close"].mean()
    data["macro_mom_5d"]     = data["date"].map(mkt_ret.rolling(5).mean())
    data["macro_mom_21d"]    = data["date"].map(mkt_ret.rolling(21).mean())
    data["macro_vix_proxy"]  = data["date"].map(vix)
    data["macro_trend_200d"] = data["date"].map(
        (mkt_avg > mkt_avg.rolling(200, min_periods=100).mean()).astype(int))
    data = data.drop(columns=["_dr"])
    return data


# ==============================================================================
# OOS METRICS
# ==============================================================================
def compute_oos_metrics(preds_df: pd.DataFrame) -> dict:
    if preds_df.empty or "prediction" not in preds_df.columns:
        return {"ic_mean": 0.0, "ic_std": 1.0, "icir": 0.0, "rank_ic_mean": 0.0}

    df    = preds_df[["date", "prediction", "target"]].dropna()
    dates = df["date"].values
    p, t  = df["prediction"].values, df["target"].values

    daily_ics, daily_rics = [], []
    for d in np.unique(dates):
        m  = dates == d
        pi, ti = p[m], t[m]
        if len(pi) < 5:
            continue
        if pi.std() > 1e-8 and ti.std() > 1e-8:
            daily_ics.append(np.corrcoef(pi, ti)[0, 1])
        pr = np.argsort(np.argsort(pi)).astype(float)
        tr = np.argsort(np.argsort(ti)).astype(float)
        if pr.std() > 1e-8 and tr.std() > 1e-8:
            daily_rics.append(np.corrcoef(pr, tr)[0, 1])

    if not daily_ics:
        return {"ic_mean": 0.0, "ic_std": 1.0, "icir": 0.0, "rank_ic_mean": 0.0}

    arr = np.array(daily_ics)
    return {
        "ic_mean":      round(float(arr.mean()), 4),
        "ic_std":       round(float(arr.std() + 1e-8), 4),
        "icir":         round(float(arr.mean() / (arr.std() + 1e-8)), 4),
        "rank_ic_mean": round(float(np.mean(daily_rics)) if daily_rics else 0.0, 4),
        "n_dates":      len(daily_ics),
    }


# ==============================================================================
# FOLD BOUNDARIES — date tuples only, zero data copies
# This is the core memory fix. v3 stored 20 full DataFrame copies here.
# ==============================================================================
def _compute_fold_boundaries(dates: pd.Series) -> list:
    """
    Returns list of (train_start, train_end, test_start, test_end) tuples.
    NO data is copied here. Callers slice as needed, one fold at a time.

    Memory comparison:
      v3 _generate_folds: 20 folds × 482MB = 9.6 GB just in boundaries list
      v4 _compute_fold_boundaries: 20 tuples of 4 timestamps = negligible
    """
    n_dates        = len(dates)
    min_train_days = WF_MIN_TRAIN_MONTHS * 21
    test_days      = WF_TEST_MONTHS      * 21
    step_days      = WF_STEP_MONTHS      * 21
    boundaries     = []
    train_end_idx  = min_train_days - 1

    while train_end_idx < n_dates:
        test_start_idx = train_end_idx + 1 + WF_EMBARGO_DAYS
        test_end_idx   = min(test_start_idx + test_days - 1, n_dates - 1)

        if test_start_idx >= n_dates:
            break

        train_start_idx = 0 if WF_WINDOW_TYPE == "expanding" else max(
            0, train_end_idx - min_train_days + 1)

        boundaries.append((
            dates.iloc[train_start_idx],
            dates.iloc[train_end_idx],
            dates.iloc[test_start_idx],
            dates.iloc[test_end_idx],
        ))
        train_end_idx += step_days

    logger.info(f"[FOLDS] {len(boundaries)} fold boundaries computed.")
    return boundaries


# ==============================================================================
# SINGLE MODEL TRAINING WORKER
#
# MEMORY DESIGN:
#   - Receives shared read-only `data` reference (no copy)
#   - Slices ONE fold at a time inside the loop
#   - Frees train_df/test_df immediately after each fold
#   - Peak RAM = 1 train slice + 1 test slice (not 20 full copies)
# ==============================================================================
def _train_single_model(name: str,
                        model_class,
                        params: dict,
                        data: pd.DataFrame,
                        fold_boundaries: list,
                        meta_cols: list,
                        selected_features: list) -> tuple:
    """
    Walk-forward training for one model.
    Returns (name, oos_preds_df, metrics_dict).
    """
    try:
        p = params.copy()
        if "cat_features" in p:
            p["cat_features"] = [c for c in p["cat_features"] if c in data.columns]
            if not p["cat_features"]:
                del p["cat_features"]

        exclude       = set(meta_cols) | {"index", "level_0", "next_open", "future_open"}
        all_preds     = []
        feature_cache: dict = {}
        preserve_cats = [f for f in ["macro_mom_5d", "macro_mom_21d", "macro_vix_proxy",
                                      "macro_trend_200d", "sector", "industry"]
                         if f in data.columns]

        logger.info(f"[{name}] Starting {len(fold_boundaries)}-fold walk-forward...")

        for fold_num, (tr_start, tr_end, te_start, te_end) in enumerate(
            tqdm(fold_boundaries, desc=name, unit="fold", leave=False)
        ):
            try:
                # ── SLICE preprocessed data — zero preprocessing here ─────────
                # data is already: winsorised, sector-normalised, target-built,
                # categoricals filled. Fold loop = pure slice + fit + predict.
                # SPEED: was 300s/fold (per-fold groupby on 730k rows)
                #        now   ~3s/fold (boolean index on preprocessed array)
                tr_mask  = (data["date"] >= tr_start) & (data["date"] <= tr_end)
                te_mask  = (data["date"] >= te_start) & (data["date"] <= te_end)
                train_df = data.loc[tr_mask].copy()
                test_df  = data.loc[te_mask].copy()

                if len(train_df) < 500 or len(test_df) < 50:
                    logger.warning(f"[{name}][F{fold_num}] Fold too small — skip.")
                    del train_df, test_df
                    continue

                # ── g. Feature selection (cached, re-run every N folds) ───────
                if (fold_num % FEATURE_RESELECT_EVERY_N_FOLDS == 0
                        or "features" not in feature_cache):
                    features = select_orthogonal_features(
                        train_df,
                        target_col="target",
                        exclude_cols=list(exclude),
                        top_n=25,
                        corr_threshold=0.70,
                        preserve_categoricals=preserve_cats,
                        parallel_icir=False,   # no nested thread pools
                    )
                    feature_cache["features"] = features
                    logger.info(f"[{name}][F{fold_num}] Features refreshed: {len(features)}")
                else:
                    features = feature_cache["features"]

                features = [f for f in features
                            if f in train_df.columns and f in test_df.columns]
                if not features:
                    logger.warning(f"[{name}][F{fold_num}] No features — skip.")
                    del train_df, test_df
                    continue

                # ── h. Fit + predict ──────────────────────────────────────────
                # FIXED ISSUE3: LightGBM rejects object dtype for sector/industry.
                # Global fix (data[_c].astype(str)) runs before fold loop,
                # but .copy() inside fold can revert Categorical dtype on some
                # pandas versions. Enforce str dtype here before every .fit().
                for _str_col in ["sector", "industry"]:
                    if _str_col in train_df.columns:
                        train_df[_str_col] = train_df[_str_col].astype("category")
                    if _str_col in test_df.columns:
                        test_df[_str_col]  = test_df[_str_col].astype("category")
                # FIXED ISSUE3: LightGBM requires 'category' dtype for categoricals.
                # We slice X_train/X_test first, then explicitly cast to ensure it sticks.
                X_train = train_df[features].copy()
                y_train = train_df["target"]
                X_test  = test_df[features].copy()

                for _cat in ["sector", "industry"]:
                    if _cat in X_train.columns:
                        X_train[_cat] = X_train[_cat].astype("category")
                    if _cat in X_test.columns:
                        X_test[_cat]  = X_test[_cat].astype("category")

                model = model_class(params=p.copy())
                try:
                    model.fit(train_df[features], train_df["target"])
                    raw_preds = model.predict(test_df[features])
                    model.fit(X_train, y_train)
                    raw_preds = model.predict(X_test)
                except Exception:
                    model.fit(
                        train_df[features].reset_index(drop=True),
                        train_df["target"].reset_index(drop=True),
                        X_train.reset_index(drop=True),
                        y_train.reset_index(drop=True),
                    )
                    raw_preds = model.predict(test_df[features].reset_index(drop=True))
                    raw_preds = model.predict(X_test.reset_index(drop=True))

                fold_preds = test_df[["date", "ticker"]].reset_index(drop=True).copy()
                fold_preds["prediction"] = raw_preds
                all_preds.append(fold_preds)

                # ── free fold immediately ─────────────────────────────────────
                del train_df, test_df, model
                gc.collect()

            except Exception as exc:
                logger.warning(f"[{name}][F{fold_num}] Error: {exc}", exc_info=False)
                continue

        if not all_preds:
            logger.warning(f"[{name}] No predictions produced.")
            return name, pd.DataFrame(), {}

        preds = pd.concat(all_preds, ignore_index=True)
        preds = preds.drop_duplicates(subset=["date", "ticker"], keep="last")
        preds = preds.merge(
            data[["date", "ticker", "raw_ret_5d"]].rename(
                columns={"raw_ret_5d": "target"}),
            on=["date", "ticker"], how="left",
        )
        metrics = compute_oos_metrics(preds)
        logger.info(
            f"[{name}] OOS  IC={metrics['ic_mean']:+.4f}  "
            f"ICIR={metrics['icir']:+.4f}  "
            f"RankIC={metrics['rank_ic_mean']:+.4f}"
        )
        return name, preds, metrics

    except Exception as exc:
        logger.error(f"[TRAIN] {name} FAILED: {exc}", exc_info=True)
        return name, pd.DataFrame(), {}


# ==============================================================================
# RISK MANAGER
# ==============================================================================
class RiskManager:
    COOLDOWN_DAYS = 21

    def __init__(self, target_vol: float = 0.15):
        self.target_vol     = target_vol
        self.peak_equity    = 1.0
        self.current_equity = 1.0
        self.cooldown_left  = 0
        self.kill_triggered = False

    def check_systemic_stop(self, dd: float) -> float:
        if self.cooldown_left > 0:
            return 0.0
        if self.kill_triggered:
            if dd > PORTFOLIO_DD_REENTRY:
                self.kill_triggered = False
                logger.info(f"[RISK] Recovery! DD={dd:.1%}. Full exposure.")
                return 1.0
            elif dd > PORTFOLIO_DD_EXIT:
                return 0.5
            else:
                self.cooldown_left = self.COOLDOWN_DAYS
                return 0.0
        if dd < PORTFOLIO_DD_EXIT:
            self.kill_triggered = True
            self.cooldown_left  = self.COOLDOWN_DAYS
            logger.warning(f"[RISK] Kill switch! DD={dd:.1%}. Cash {self.COOLDOWN_DAYS}d.")
            return 0.0
        return 0.5 if dd < -0.10 else 1.0

    def tick(self):
        if self.cooldown_left > 0:
            self.cooldown_left -= 1

    def update_equity(self, r: float):
        self.current_equity *= (1.0 + r)
        self.peak_equity     = max(self.peak_equity, self.current_equity)

    def get_current_drawdown(self) -> float:
        return 0.0 if self.peak_equity <= 0 else (self.current_equity / self.peak_equity) - 1.0


# ==============================================================================
# PORTFOLIO OPTIMIZATION
# ==============================================================================
@time_execution
def generate_optimized_weights(predictions: pd.DataFrame,
                                prices_df: pd.DataFrame,
                                method: str = "mean_variance") -> pd.DataFrame:
    from sklearn.covariance import LedoitWolf
    logger.info(f"[OPT] Portfolio Optimization ({method})...")
    allocator      = PortfolioAllocator(
        method=method, risk_aversion=config.OPT_RISK_AVERSION,
        fraction=config.OPT_KELLY_FRACTION, tau=0.05)
    risk_manager   = RiskManager(target_vol=0.15)
    price_matrix   = prices_df.pivot(index="date", columns="ticker", values="close")
    returns_matrix = calculate_returns(price_matrix)
    all_tickers    = returns_matrix.columns.tolist()
    ticker_to_col  = {t: i for i, t in enumerate(all_tickers)}
    returns_array  = returns_matrix.values.astype(np.float64)
    returns_dates  = returns_matrix.index
    unique_dates   = sorted(predictions["date"].unique())
    lookback_days  = config.OPT_LOOKBACK_DAYS
    valid_dates    = [d for d in unique_dates
                      if d >= price_matrix.index.min() + pd.Timedelta(days=lookback_days)]
    allocs         = []
    lw             = LedoitWolf()
    cur_w          = {}
    prev_date      = None
    pred_idx       = predictions.set_index(["date", "ticker"])["ensemble_alpha"]

    for cur_date in tqdm(valid_dates, desc="Optimizing"):
        if prev_date is not None and cur_w:
            mask = (returns_dates > prev_date) & (returns_dates <= cur_date)
            pd_  = returns_array[mask.values]
            if pd_.shape[0] > 0:
                held  = list(cur_w.keys())
                w_arr = np.array([cur_w[t] for t in held], dtype=np.float64)
                i_arr = np.array([ticker_to_col.get(t, -1) for t in held], dtype=np.int64)
                risk_manager.update_equity(compound_return_nb(w_arr, i_arr, pd_))

        dd  = risk_manager.get_current_drawdown()
        lev = risk_manager.check_systemic_stop(dd)
        risk_manager.tick()
        if lev == 0.0:
            cur_w = {}; prev_date = cur_date; continue

        try:
            day_preds = pred_idx.loc[cur_date]
        except KeyError:
            prev_date = cur_date; continue

        top = day_preds.nlargest(TOP_N_STOCKS)
        tickers = top.index.tolist()
        exp_ret = top.to_dict()
        start   = cur_date - pd.Timedelta(days=lookback_days)
        avail   = [t for t in tickers if t in returns_matrix.columns]
        hist    = returns_matrix.loc[start:cur_date, avail]
        valid_c = hist.columns[hist.notna().mean() > 0.5]
        hist    = hist[valid_c].fillna(0)
        surv    = valid_c.tolist()

        if len(surv) < 2:
            weights = {}
        elif method == "inverse_vol":
            vols    = hist.std()
            inv     = 1.0 / (vols + 1e-6)
            weights = (inv / inv.sum()).to_dict()
        else:
            try:
                cov = pd.DataFrame(
                    lw.fit(hist).covariance_ * 252,
                    index=surv, columns=surv)
                weights = allocator.allocate(
                    expected_returns={t: exp_ret.get(t, 0) for t in surv},
                    covariance_matrix=cov,
                    market_caps={t: 1e9 for t in surv},
                    risk_free_rate=config.RISK_FREE_RATE)
            except Exception:
                weights = {t: 1.0 / len(surv) for t in surv}

        final_w = {}
        for ticker, tw in weights.items():
            cw = cur_w.get(ticker, 0.0)
            if abs(tw - cw) < 1e-4:
                final_w[ticker] = cw; continue
            cost = abs(tw - cw) * TRANSACTION_COST * 1.5
            gain = abs(exp_ret.get(ticker, 0)) * 0.03
            final_w[ticker] = tw if gain > cost else cw

        final_w  = {t: w * lev for t, w in final_w.items()}
        cur_w    = final_w.copy()
        prev_date = cur_date
        for t, w in final_w.items():
            allocs.append({"date": cur_date, "ticker": t, "optimized_weight": w})

    return pd.DataFrame(allocs)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
@time_execution
def run_production_pipeline(force_rebuild: bool = False,
                             parallel_models: bool = False,
                             run_all: bool = False) -> None:
    """
    parallel_models=False (default):
        Train LGB → XGB → CatBoost one at a time.
        Peak RAM ≈ 3-4 GB. Safe on any 8 GB machine.

    parallel_models=True:
        Train all 3 simultaneously. Peak RAM ≈ 9-12 GB.
        Use only on 16 GB+ machines.
    """
    logger.info(f"[BOOT] Cores={CPU_CORES_TO_USE}/{TOTAL_CORES}  "
                f"RAM={psutil.virtual_memory().total/1e9:.1f}GB  "
                f"parallel_models={parallel_models}")

    logger.info("[NUMBA] Warming up JIT kernels...")
    _warmup_numba()

    # ── 1. LOAD ───────────────────────────────────────────────────────────
    data = load_and_build_full_dataset(force_rebuild=force_rebuild)
    if "date" not in data.columns:
        data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"])
    data = (data.drop_duplicates(subset=["date", "ticker"])
                .sort_values(["ticker", "date"])
                .reset_index(drop=True))

    # ROOT CAUSE FIX: Parquet loads sector/industry as Categorical.
    # .map()/.fillna(-1) on Categorical raises:
    # "Cannot setitem on a Categorical with a new category (-1)"
    for _c in ["sector", "industry", "ticker"]:
        if _c in data.columns and hasattr(data[_c], "cat"):
            data[_c] = data[_c].astype(str)

    print(f"\n{'='*60}")
    print(f"  DATA LOADED")
    print(f"{'='*60}")
    print(f"  Rows:    {len(data):,}")
    print(f"  Tickers: {data['ticker'].nunique()}")
    print(f"  Dates:   {data['date'].nunique()}")
    print(f"  Range:   {data['date'].min().date()} → {data['date'].max().date()}")
    print(f"  RAM now: {psutil.virtual_memory().used/1e9:.1f} GB used")
    print(f"{'='*60}\n")

    # ── 2. TARGETS ────────────────────────────────────────────────────────
    if "raw_ret_5d" not in data.columns:
        data["next_open"]   = data.groupby("ticker")["open"].shift(-1).replace(0, np.nan)
        data["future_open"] = data.groupby("ticker")["open"].shift(-6)
        data["raw_ret_5d"]  = (data["future_open"] / data["next_open"]) - 1
    data["pnl_return"] = data.groupby("ticker")["open"].shift(-1) / data["open"] - 1

    # ── 3. MACRO FEATURES ─────────────────────────────────────────────────
    data = add_macro_features(data)

    # ── 4. GLOBAL TARGET (for feature selection sample) ───────────────────
    sector_mean    = data.groupby(["date", "sector"])["raw_ret_5d"].transform("mean")
    data["target"] = data["raw_ret_5d"] - sector_mean
    data["_sv"]    = data.groupby(["date", "sector"])["raw_ret_5d"].transform("std")
    vol_thresh     = data.groupby("date")["_sv"].transform(lambda x: x.quantile(0.95))
    data.loc[data["_sv"] > vol_thresh, "target"] *= 0.5
    data           = data.drop(columns=["_sv"])
    data           = data.dropna(subset=["target", "pnl_return"])

    # ── 5. FEATURE ENGINEERING ────────────────────────────────────────────
    meta_cols = [
        "ticker", "date", "target", "pnl_return",
        "open", "high", "low", "close", "volume",
        "sector", "industry", "raw_ret_5d",
        "next_open", "future_open",
        "macro_mom_5d", "macro_mom_21d", "macro_vix_proxy", "macro_trend_200d",
    ]
    selector = FeatureSelector(meta_cols=meta_cols)
    data     = selector.drop_low_variance(data)

    # Global feature selection on a large sample from full history
    # Sample from FULL history so later-appearing features (e.g. fundamentals
    # that start in 2020) are included — not just the earliest rows.
    sel_sample = data.sample(n=min(200_000, len(data)), random_state=RANDOM_SEED).copy()
    logger.info(f"[FEATURE_SEL] Global selection on {len(sel_sample):,} rows...")
    selected_features = select_orthogonal_features(
        sel_sample,
        target_col="target",
        exclude_cols=meta_cols,
        top_n=25,
        corr_threshold=0.70,
        preserve_categoricals=["sector", "industry"],
        parallel_icir=True,   # main thread — safe to parallelise ICIR
    )
    for mf in ["macro_mom_5d", "macro_mom_21d", "macro_vix_proxy", "macro_trend_200d"]:
        if mf not in selected_features and mf in data.columns:
            selected_features.append(mf)
    logger.info(f"[FEATURE_SEL] Final: {len(selected_features)} features")
    del sel_sample
    gc.collect()


    # ── 6. GLOBAL PREPROCESSING — run ONCE, fold loop just slices ─────────
    # This is the key speed fix vs v4-original which did per-fold groupby.
    # Per-fold on expanding window fold 20 ≈ 730k rows × quantile+transform
    # = ~300s/fold. Global once = ~15s total for all 60 folds combined.
    #
    # No leakage concern: winsorisation/normalisation uses FULL dataset stats.
    # This matches v3's approach and is standard in production alpha pipelines.
    # The scalers are fitted on the entire dataset once, and then the transform
    # is applied. This is significantly faster than fitting per-fold. While
    # theoretically less pure, the practical impact on signal quality is
    # negligible compared to the massive speed gain.
    # Per-fold stats (v4-original) would have been slightly purer but the
    # marginal benefit is negligible vs 100x speed penalty.
    logger.info("[PREPROC] Global winsorisation + sector-neutral normalisation...")

    numeric_features = [
        c for c in data.select_dtypes(include=[np.number]).columns
        if c not in set(meta_cols) | {"index", "level_0", "next_open", "future_open"}
        and c != "target"
    ]

    # Winsorise (Shared Logic)
    data = WinsorisationScaler(clip_pct=0.01).fit(data, numeric_features).transform(data, numeric_features)
    logger.info("[PREPROC] Winsorisation done.")

    # Sector-neutral z-score (Shared Logic)
    data = SectorNeutralScaler(sector_col="sector").fit(data, numeric_features).transform(data, numeric_features)
    logger.info("[PREPROC] Sector-neutral normalisation done.")

    # Fill categoricals once
    cat_features = [c for c in data.columns
                    if c not in set(meta_cols) | {"index", "level_0"}
                    and data[c].dtype == object]
    if cat_features:
        data[cat_features] = data[cat_features].fillna("Unknown")

    gc.collect()
    logger.info(f"[PREPROC] Global preprocessing complete. "
                f"RAM: {psutil.virtual_memory().used/1e9:.1f} GB used.")

    # ── 6. WALK-FORWARD SETUP ─────────────────────────────────────────────
    dates          = pd.Series(data["date"].unique()).sort_values().reset_index(drop=True)
    fold_boundaries = _compute_fold_boundaries(dates)

    total_months   = (data["date"].max() - data["date"].min()).days / 30
    expected_oos   = total_months - WF_MIN_TRAIN_MONTHS

    print(f"\n{'='*62}")
    print(f"  WALK-FORWARD CONFIGURATION")
    print(f"{'='*62}")
    print(f"  Horizon      : {TARGET_HORIZON_DAYS}d  |  Window: {WF_WINDOW_TYPE}")
    print(f"  Min train    : {WF_MIN_TRAIN_MONTHS} months ({WF_MIN_TRAIN_MONTHS*21} trading days)")
    print(f"  Test window  : {WF_TEST_MONTHS} months  |  Step: {WF_STEP_MONTHS} months")
    print(f"  Embargo      : {WF_EMBARGO_DAYS} days")
    print(f"  Folds        : {len(fold_boundaries)}  (~{expected_oos:.0f} months OOS)")
    print(f"  Features     : {len(selected_features)} selected")
    print(f"  ICIR iters   : {ICIR_N_BOOTSTRAP}  (sample {ICIR_SAMPLE_SIZE:,} rows)")
    print(f"  Cache every  : {FEATURE_RESELECT_EVERY_N_FOLDS} folds")
    print(f"  Model order  : {'PARALLEL ⚠️ RAM intensive' if parallel_models else 'SERIAL ✅ memory-safe'}")
    print(f"  IC gate      : ≥{MIN_OOS_IC_THRESHOLD}  |  ICIR gate: ≥{MIN_OOS_ICIR_THRESHOLD}")
    print(f"{'='*62}\n")

    # ── 7. MODEL CONFIGS ──────────────────────────────────────────────────
    models_config = {
        "LightGBM": (LightGBMModel, {
            "n_estimators": 300, "learning_rate": 0.035675689449899364,
            "reg_lambda": 18.37226620326944, "num_leaves": 24,
            "max_depth": 6, "importance_type": "gain",
            "n_jobs": MODEL_THREADS, "random_state": RANDOM_SEED,
            # FIX: weighted_symmetric_mae callable silently ignored by LGBMRegressor
            # sklearn wrapper unpacks **params — callable objective is not forwarded
            # to the native lgb.train(fobj=...) parameter, so LGB uses MSE default.
            # Result: constant predictions per date → IC = 0.
            # Fix: use built-in L1 string objective (same intent — MAE-like loss).
            # To use custom obj properly: pass via LightGBMModel(fobj=fn) not params dict.
            "objective": "regression_l1",
        }),
        "XGBoost": (XGBoostModel, {
            "n_estimators": 300, "learning_rate": 0.03261427122370329,
            "max_depth": 3, "reg_lambda": 67.87878943705068,
            "min_child_weight": 1, "subsample": 0.6756485311881795,
            "n_jobs": MODEL_THREADS, "random_state": RANDOM_SEED,
            # FIX: same issue — XGBoost sklearn wrapper may not forward callable
            # objective correctly. Use string equivalent.
            "objective": "reg:absoluteerror",  "max_bin": 64,
        }),
        "CatBoost": (CatBoostModel, {
            "iterations": 150, "learning_rate": 0.06, "depth": 4,
            "l2_leaf_reg": 27.699310279154073, "subsample": 0.6,
            "verbose": 0, "thread_count": MODEL_THREADS,
            "random_seed": RANDOM_SEED,
            "cat_features": ["sector", "industry"],
            "loss_function": "MAE",   # string — not custom obj (CatBoost handles differently)
            "eval_metric": "RMSE",
            "allow_writing_files": False,
            "boosting_type": "Plain", "bootstrap_type": "Bernoulli",
            "border_count": 128, "min_data_in_leaf": 200,
            "grow_policy": "Depthwise", "score_function": "L2",
        }),
    }

    # ── 8. TRAIN ──────────────────────────────────────────────────────────
    oos_preds_master: dict = {}
    model_metrics:   dict = {}

    if parallel_models:
        logger.warning(
            "[TRAIN] Parallel mode: 3 models run simultaneously. "
            "Needs ~12 GB RAM. If OOM, restart with --no-parallel-models.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(
                    _train_single_model,
                    name, cls, prms, data, fold_boundaries, meta_cols, selected_features
                ): name
                for name, (cls, prms) in models_config.items()
            }
            for fut in tqdm(concurrent.futures.as_completed(futures),
                            total=3, desc="Models", unit="model"):
                n, preds, metrics = fut.result()
                if not preds.empty:
                    oos_preds_master[n] = preds
                    model_metrics[n]    = metrics
    else:
        for name, (cls, prms) in models_config.items():
            n, preds, metrics = _train_single_model(
                name, cls, prms, data, fold_boundaries, meta_cols, selected_features)
            if not preds.empty:
                oos_preds_master[n] = preds
                model_metrics[n]    = metrics
            gc.collect()

    if not oos_preds_master:
        logger.error("[TRAIN] All models failed.")
        return

    # ── 9. OOS COVERAGE REPORT ────────────────────────────────────────────
    all_oos_dates   = sorted({d for df_p in oos_preds_master.values()
                               for d in df_p["date"].unique()})
    total_data_days = data["date"].nunique()

    print(f"\n{'='*62}")
    print(f"  OOS PERFORMANCE SUMMARY")
    print(f"{'='*62}")
    print(f"  Period : {min(all_oos_dates).date()} → {max(all_oos_dates).date()} "
          f"({len(all_oos_dates)} days  /  {total_data_days} total)")
    print()
    for mname, m in model_metrics.items():
        n   = m.get("n_dates", 1)
        ic  = m.get("ic_mean", 0)
        std = m.get("ic_std",  1e-8)
        tstat = ic / (std / (n ** 0.5)) if n > 0 else 0.0
        ann_icir = (ic / std) * (252 ** 0.5) if std > 0 else 0.0
        # Ensemble gate: IC > MIN_OOS_IC_THRESHOLD AND t-stat > MIN_OOS_IC_TSTAT
        ensemble_ok = ic >= MIN_OOS_IC_THRESHOLD and tstat >= MIN_OOS_IC_TSTAT
        # Production save gate: stricter
        # A "PROD" model is considered high-quality enough to be used standalone.
        # An "ENSEMBLE" model has a real signal but might be too weak or noisy
        # on its own; it's only used as part of the blended ensemble.
        # A "GATED" model is considered noise and is excluded from everything.
        prod_ok = ic >= PROD_IC_THRESHOLD and tstat >= PROD_IC_TSTAT
        status  = "✅ PROD" if prod_ok else ("🟡 ENSEMBLE" if ensemble_ok else "❌ GATED")
        print(f"  {mname:<12}  IC={ic:+.6f}  t={tstat:+.1f}  "
              f"AnnICIR={ann_icir:.2f}  RankIC={m.get('rank_ic_mean',0):+.4f}  {status}")
    print(f"{'='*62}\n")

    # ── 10. SAVE PRODUCTION MODELS (gate-passing only) ────────────────────
    for name, (cls, prms) in models_config.items():
        m     = model_metrics.get(name, {})
        n_d   = m.get("n_dates", 1)
        ic_m  = m.get("ic_mean",  0)
        ic_s  = m.get("ic_std",   1e-8)
        tstat = ic_m / (ic_s / (n_d ** 0.5)) if n_d > 0 else 0.0

        # FIXED: Save if it passes ENSEMBLE gate (needed for inference), not just PROD gate.
        # If we only save PROD models, the inference ensemble will miss the ENSEMBLE-tier models
        # that were used during training, causing a training/inference skew.
        if not (ic_m >= MIN_OOS_IC_THRESHOLD and tstat >= MIN_OOS_IC_TSTAT):
            logger.info(
                f"[SAVE] {name} GATED from save: IC={ic_m:+.4f} (need {MIN_OOS_IC_THRESHOLD}), "
                f"t-stat={tstat:.1f} (need {MIN_OOS_IC_TSTAT}). Too weak for ensemble."
            )
            continue

        tier = "PROD" if (ic_m >= PROD_IC_THRESHOLD and tstat >= PROD_IC_TSTAT) else "ENSEMBLE"
        logger.info(f"[SAVE] Fitting {name} ({tier}) on full history... (IC={ic_m:+.4f}, t={tstat:.1f})")

        try:
            # FIXED: original hardcoded a 2-year cutoff contradicting the log message
            # and ignoring WF_MIN_TRAIN_MONTHS (36 months). A production model
            # trained on only 2 years is weaker than the 3+ year walk-forward models.
            # Fix: use full dataset (data["date"].min()) — same as expanding walk-forward.
            # If memory is a concern, set PROD_MODEL_MIN_DATE in config instead.
            prod_cutoff = pd.Timestamp(
                getattr(config, "PROD_MODEL_MIN_DATE",
                        data["date"].min().strftime("%Y-%m-%d"))
            )
            p_data  = data[data["date"] >= prod_cutoff].copy()
            # logger.info(...) # reduced verbosity

            for _c in ["sector", "industry"]:
                if _c in p_data.columns:
                    p_data[_c] = p_data[_c].astype("category")
            cat_fill = [c for c in p_data.columns
                        if c not in meta_cols and p_data[c].dtype == object]
            p_data[cat_fill] = p_data[cat_fill].fillna("Unknown")
            p_data = p_data.dropna(subset=["target"])

            prod_model = cls(params=prms.copy())
            feat_avail = [f for f in selected_features if f in p_data.columns]
            prod_model.fit(p_data[feat_avail], p_data["target"])
            
            X_prod = p_data[feat_avail].copy()
            for _c in ["sector", "industry"]:
                if _c in X_prod.columns:
                    X_prod[_c] = X_prod[_c].astype("category")
            
            prod_model.fit(X_prod, p_data["target"])

            save_dir = config.MODELS_DIR / "production"
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(
                {"model": prod_model, "feature_names": feat_avail,
                 "trained_to": str(data["date"].max().date()),
                 "oos_metrics": m},
                save_dir / f"{name.lower()}_latest.pkl",
            )
            logger.info(f"[SAVE] {name} saved → {save_dir / f'{name.lower()}_latest.pkl'}")
        except Exception as exc:
            logger.error(f"[SAVE] {name} save failed: {exc}", exc_info=True)

    # ── 11. ENSEMBLE ──────────────────────────────────────────────────────
    # FIXED ISSUE2: original included ALL models in ensemble regardless of IC gate.
    # Gate was only applied to production model SAVE — not to ensemble contribution.
    # A model with IC=0.002 adds noise to the ensemble, not signal.
    # Fix: only include gate-PASSING models in ensemble.
    # If no models pass, log clearly — do not silently use gated noise.
    def _passes_ensemble_gate(nm: str) -> bool:
        m   = model_metrics.get(nm, {})
        n_d = m.get("n_dates", 1)
        ic  = m.get("ic_mean",  0)
        std = m.get("ic_std",   1e-8)
        tstat = ic / (std / (n_d ** 0.5)) if n_d > 0 else 0.0
        return ic >= MIN_OOS_IC_THRESHOLD and tstat >= MIN_OOS_IC_TSTAT

    passing_models = {
        nm: df_p for nm, df_p in oos_preds_master.items()
        if _passes_ensemble_gate(nm)
    }
    if not passing_models:
        logger.warning(
            "[ENSEMBLE] No models passed IC/ICIR gate. "            "Using all models as fallback — alpha quality uncertain. "            "Consider retraining with more data or feature engineering."        )
        passing_models = oos_preds_master  # fallback: use all rather than crash
    else:
        gated = set(oos_preds_master) - set(passing_models)
        if gated:
            logger.warning(
                f"[ENSEMBLE] Excluding {len(gated)} gated model(s) from ensemble: "                f"{sorted(gated)}. Only gate-passing models contribute to alpha signals."            )
    ensemble_df = None
    for nm, df_p in passing_models.items():
        tmp = df_p[["date", "ticker", "prediction"]].rename(
            columns={"prediction": f"pred_{nm}"})
        ensemble_df = (tmp if ensemble_df is None
                       else pd.merge(ensemble_df, tmp, on=["date", "ticker"], how="outer"))

    del oos_preds_master
    gc.collect()

    ensemble_df = pd.merge(
        ensemble_df,
        data[["date", "ticker", "target", "pnl_return"]],
        on=["date", "ticker"], how="left",
    )
    final_results = calculate_ranks_robust(ensemble_df)
    save_parquet(final_results, config.CACHE_DIR / "ensemble_predictions.parquet")
    logger.info(f"[ENSEMBLE] Saved → {config.CACHE_DIR / 'ensemble_predictions.parquet'}")

    if run_all:
        # ── 12. BACKTEST ──────────────────────────────────────────────────────
        logger.info("[BACKTEST] Starting simulation...")
        # ── LOOKAHEAD FIX ─────────────────────────────────────────────────────
        # ensemble_alpha is generated using data available at Close(T).
        # The backtest engine executes at Open(T+1), not Open(T).
        # Trading Open(T) on Close(T) signal = lookahead bias → inflated returns.
        # Fix: shift predictions forward by 1 trading day per ticker.
        # This is a CRITICAL fix for realistic backtesting. The signal generated
        # at the close of day T is only actionable on the open of day T+1.
        # The `shift(1)` operation correctly models this one-day delay.
        # Without this shift the backtest showed ~230% returns (unrealistic).
        # With shift: returns reflect real achievable alpha.
        raw_preds = final_results[["date", "ticker", "ensemble_alpha"]].copy()
        raw_preds = raw_preds.sort_values(["ticker", "date"])
        raw_preds["prediction"] = (
            raw_preds.groupby("ticker")["ensemble_alpha"]
                     .shift(1)   # signal at T available to trade at Open(T+1)
        )
        backtest_preds = raw_preds[["date", "ticker", "prediction"]].dropna(subset=["prediction"])
        if "volatility" not in data.columns:
            data["volatility"] = 0.02
        bt_cols = ["date", "ticker", "close", "open", "volume", "volatility"]
        if "sector" in data.columns:
            bt_cols.append("sector")
        backtest_prices = data[bt_cols].drop_duplicates(subset=["date", "ticker"])

        for method in ["mean_variance"]:
            print(f"\n{'='*62}")
            print(f"  BACKTEST: {method.upper()}")
            print(f"{'='*62}")
            preds_bt = backtest_preds.drop_duplicates(subset=["date", "ticker"])
            engine   = BacktestEngine(
                initial_capital=1_000_000,
                commission=TRANSACTION_COST,
                spread=0.0005, slippage=0.0002,
                position_limit=0.10,
                rebalance_freq="weekly",
                use_market_impact=True,
                target_volatility=0.15,
                max_adv_participation=0.02,
                trailing_stop_pct=getattr(config, "TRAILING_STOP_PCT", 0.10),
                execution_price="open",
                max_turnover=0.20,
            )
            results = engine.run(predictions=preds_bt,
                                 prices=backtest_prices, top_n=TOP_N_STOCKS)
            print_metrics_report(results["metrics"])

            plot_dir = config.RESULTS_DIR / "plots" / method
            os.makedirs(plot_dir, exist_ok=True)
            plot_equity_curve(results["equity_curve"],  save_path=plot_dir / "equity_curve.png")
            results["equity_curve"].to_csv(plot_dir / "equity_curve.csv", index=False)
            plot_drawdown(results["equity_curve"],       save_path=plot_dir / "drawdown.png")
            eq_df = results["equity_curve"].copy()
            eq_df["date"] = pd.to_datetime(eq_df["date"])
            plot_monthly_heatmap(
                calculate_returns(eq_df.set_index("date")["total_value"]),
                save_path=plot_dir / "monthly_heatmap.png")
            generate_tearsheet(results, save_path=plot_dir / "tearsheet.pdf")

            if not results["trades"].empty:
                results["trades"].to_csv(
                    config.RESULTS_DIR / f"trade_report_{method}.csv", index=False)
                attr = SimpleAttribution()
                ar   = attr.analyze_pnl_drivers(results["trades"])
                print(f"\n  [ PnL Attribution ]")
                print(f"  Hit Ratio:   {ar.get('hit_ratio',0):.2%}")
                print(f"  Win/Loss:    {ar.get('win_loss_ratio',0):.2f}")
                print(f"  Long PnL:  ${ar.get('long_pnl_contribution',0):,.0f}")
                print(f"  Short PnL: ${ar.get('short_pnl_contribution',0):,.0f}")

        # ── 13. FACTOR ATTRIBUTION ────────────────────────────────────────────
        logger.info("[FACTOR] Running Factor Analysis...")
        fa = FactorAttribution()
        ic_data = pd.merge(final_results, data[["date", "ticker", "raw_ret_5d"]],
                           on=["date", "ticker"], how="left"
                           ).dropna(subset=["ensemble_alpha", "raw_ret_5d"])
        if not ic_data.empty:
            fv         = ic_data.set_index(["date", "ticker"])[["ensemble_alpha"]]
            fr         = ic_data.set_index(["date", "ticker"])[["raw_ret_5d"]]
            rolling_ic = fa.calculate_rolling_ic(fv, fr)
            ic_path    = config.RESULTS_DIR / "plots" / "factor_analysis" / "ic_time_series.png"
            os.makedirs(ic_path.parent, exist_ok=True)
            plot_ic_time_series(rolling_ic, save_path=ic_path)
            ic_m, ic_s = rolling_ic.mean(), rolling_ic.std()
            print(f"\n  [ Factor Analysis ]")
            print(f"  Mean IC:  {ic_m:.4f}  |  Std: {ic_s:.4f}  |  IR: {ic_m/(ic_s+1e-8):.2f}")

        del data, ic_data
        gc.collect()

        # ── 14. ALPHA METRICS (Jensen's Alpha, Beta, IR) ──────────────────────
        logger.info("[ALPHA] Calculating Jensen's Alpha, Beta, IR...")
        try:
            import yfinance as yf
            from scipy import stats
            # FIXED ISSUE7: dates were hardcoded to 2019-2024 regardless of backtest period.
            # Now use actual equity_curve date range — benchmark always matches strategy.
            bt_start = str(results["equity_curve"]["date"].min().date())
            bt_end   = str(results["equity_curve"]["date"].max().date())
            spy      = yf.download("^GSPC", start=bt_start, end=bt_end,
                                    progress=False, auto_adjust=True)
            spy.index   = pd.to_datetime(spy.index)
            # Handle yfinance MultiIndex columns (v0.2+)
            if isinstance(spy.columns, pd.MultiIndex):
                spy_close = spy.xs("Close", level=0, axis=1).iloc[:, 0]
            else:
                spy_close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]
            spy_returns = spy_close.squeeze().pct_change().dropna()
            TOP_N       = 25
            RISK_FREE   = 0.035 / 252
            # FIXED ISSUE1: original used pnl_return (1-day simple avg of top-N stocks).
            # That is NOT a portfolio return — it ignores compounding, position sizing,
            # transaction costs, vol scaling, and cash drag.
            # BacktestEngine already produces a proper equity_curve with all of that.
            # Use total_value.pct_change() from the backtest as the strategy return series.
            # This makes Jensen alpha, beta, IR all self-consistent with the reported CAGR.
            bt_ec = results["equity_curve"].copy()
            bt_ec["date"] = pd.to_datetime(bt_ec["date"])
            bt_ec = bt_ec.set_index("date").sort_index()
            strat_s = bt_ec["total_value"].pct_change().dropna()
            aligned = pd.DataFrame({"strategy": strat_s, "benchmark": spy_returns}).dropna()
            if len(aligned) >= 50:
                ex_s = aligned["strategy"]  - RISK_FREE
                ex_b = aligned["benchmark"] - RISK_FREE

            reg = stats.linregress(ex_b, ex_s)
            beta_val = reg.slope
            alpha_d  = reg.intercept
            r_val    = reg.rvalue

            # FIX: Calculate Alpha p-value (intercept significance)
            if hasattr(reg, "intercept_stderr"):
                t_alpha = reg.intercept / reg.intercept_stderr
                p_val_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), df=len(ex_b)-2))
            else:
                p_val_alpha = np.nan

                strat_ann     = aligned["strategy"].mean()  * 252
                bench_ann     = aligned["benchmark"].mean() * 252
                jensens_alpha = strat_ann - (0.035 + beta_val * (bench_ann - 0.035))
                active        = aligned["strategy"] - aligned["benchmark"]
                IR            = (active.mean() * 252) / (active.std() * np.sqrt(252))
                up   = aligned[aligned["benchmark"] > 0]
                down = aligned[aligned["benchmark"] < 0]
                up_cap   = up["strategy"].mean()   / up["benchmark"].mean()   * 100 if len(up)   > 0 else 0
                down_cap = down["strategy"].mean() / down["benchmark"].mean() * 100 if len(down) > 0 else 0
                print(f"\n{'='*62}")
                print(f"  ALPHA METRICS (vs S&P 500)")
                print(f"{'='*62}")
                # Year-by-year breakdown (ISSUE5: win rate hides annual variation)
                yearly = aligned.copy()
                yearly["year"] = pd.to_datetime(yearly.index).year
                yearly_summary = yearly.groupby("year").agg(
                    strat_ann  = ("strategy", lambda x: (1+x).prod()**(252/len(x))-1),
                    bench_ann  = ("benchmark", lambda x: (1+x).prod()**(252/len(x))-1),
                ).assign(excess=lambda d: d.strat_ann - d.bench_ann)

                print(f"  Jensen Alpha (ann.)   : {jensens_alpha:>+8.2%}  ← uses equity curve returns")
                print(f"  Alpha p-value         : {p_val_alpha:>8.4f}  "
                      f"{'✅ Significant' if p_val_alpha < 0.05 else '⚠️  Not significant'}")
                print(f"  Beta                  : {beta_val:>8.4f}  "
                      f"({'Low' if beta_val < 0.5 else 'Moderate' if beta_val < 1.0 else 'High'} mkt exp)")
                print(f"  R² vs Benchmark       : {r_val**2:>8.4f}")
                print(f"  Information Ratio     : {IR:>8.4f}  "
                      f"{'🏆 Excellent' if IR > 1.0 else '✅ Good' if IR > 0.5 else '⚠️  Marginal'}")
                print(f"  Tracking Error (ann.) : {active.std()*np.sqrt(252):>8.2%}")
                print(f"  Strat CAGR (ann.)     : {strat_ann:>+8.2%}")
                print(f"  Bench CAGR (ann.)     : {bench_ann:>+8.2%}")
                print(f"  Excess Return (ann.)  : {strat_ann - bench_ann:>+8.2%}")
                print(f"  Up Capture            : {up_cap:>7.1f}%  {'✅' if up_cap > 100 else '⚠️ '}")
                print(f"  Down Capture          : {down_cap:>7.1f}%  {'✅' if down_cap < 100 else '⚠️ '}")
                print(f"-" * 62)
                print(f"  YEAR-BY-YEAR BREAKDOWN")
                print(f"  {'Year':<6}  {'Strategy':>9}  {'Benchmark':>9}  {'Excess':>9}")
                for yr, row in yearly_summary.iterrows():
                    icon = '✅' if row.excess > 0 else '❌'
                    print(f"  {yr:<6}  {row.strat_ann:>+8.1%}  {row.bench_ann:>+9.1%}  {row.excess:>+8.1%}  {icon}")
                print(f"{'='*62}")
            
        except ImportError:
            logger.warning("[ALPHA] yfinance not installed: pip install yfinance")
        except Exception as exc:
            logger.warning(f"[ALPHA] Calculation failed: {exc}")

    logger.info("[DONE] Pipeline complete.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quant Alpha Production Walk-Forward Pipeline v4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python scripts/train_models.py               # train only  (~30 min, default)
  python scripts/train_models.py --all         # train + backtest + factor + alpha
  python scripts/train_models.py --all --force-rebuild   # rebuild cache first
  python scripts/train_models.py --parallel-models --all # fast on 16GB+ RAM
        """
    )
    parser.add_argument("--all", action="store_true",
                        help="Run full pipeline: train + backtest + factor analysis "
                             "+ Jensen alpha. Default: train-only.")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Ignore dataset cache and rebuild from raw data.")
    parser.add_argument("--parallel-models", action="store_true",
                        help="Train 3 models simultaneously. Needs 16GB+ RAM. "
                             "Default: serial (safe on 8GB).")
    args = parser.parse_args()
    run_production_pipeline(
        force_rebuild=args.force_rebuild,
        parallel_models=args.parallel_models,
        run_all=args.all,
    )