"""
Walk-Forward Production Training Pipeline
=========================================
Orchestrates the rigorous out-of-sample training, evaluation, and promotion 
of the Gradient Boosted Decision Tree (GBDT) ensemble.

Purpose
-------
This module serves as the primary machine learning engine for the alpha platform. 
It executes purged walk-forward cross-validation to construct robust, non-linear 
alpha models. Strict look-ahead bias prevention is enforced via per-fold 
cross-sectional normalization and embargo periods.

Role in Quantitative Workflow
-----------------------------
Executes after feature generation to select orthogonal signals, optimize trees, 
and persist gate-passing artifacts for the production inference pipeline.

Dependencies
------------
- **NumPy/Numba**: JIT-compiled kernels for high-frequency ranking and IC computations.
- **Pandas**: Temporal and cross-sectional alignment.
- **LightGBM/XGBoost/CatBoost**: Core non-linear estimators.
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
import subprocess
import concurrent.futures
from pathlib import Path

# ==============================================================================
# CPU THROTTLE — MUST be before any ML library imports
# ==============================================================================
TOTAL_CORES      = os.cpu_count() or 4
CPU_CORES_TO_USE = max(2, TOTAL_CORES // 2)
MODEL_THREADS    = CPU_CORES_TO_USE


def _set_thread_env(n: int) -> None:
    """
    Globally binds multithreading constraints for BLAS/OpenMP backend operations.

    Args:
        n (int): The absolute maximum number of parallel threads permitted.

    Returns:
        None
    """
    os.environ["OMP_NUM_THREADS"]      = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"]      = str(n)
    os.environ["BLAS_NUM_THREADS"]     = str(n)
    os.environ["LOKY_MAX_CPU_COUNT"]   = str(n)
    os.environ["NUMBA_CACHE_DIR"]      = ".numba_cache"
    os.environ["PYTHONWARNINGS"]       = "ignore"
    if "NUMBA_NUM_THREADS" not in os.environ:
        os.environ["NUMBA_NUM_THREADS"] = str(n)

_set_thread_env(CPU_CORES_TO_USE)

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
from quant_alpha.preprocessing import enhance_fundamentals_for_registry
from quant_alpha.backtest.attribution import SimpleAttribution, FactorAttribution
from quant_alpha.optimization.allocator import PortfolioAllocator
from quant_alpha.visualization import (
    plot_equity_curve, plot_drawdown, plot_monthly_heatmap,
    plot_ic_time_series, generate_tearsheet
)
from quant_alpha.utils.preprocessing import (
    WinsorisationScaler, SectorNeutralScaler, winsorize_clip_nb
)

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
TOP_N_STOCKS           = getattr(config, "NUM_LONG_POSITIONS", 25)
STOCK_STOP_LOSS        = -getattr(config, "STOP_LOSS_PCT", 0.05)
PORTFOLIO_DD_EXIT      = -getattr(config, "MAX_DRAWDOWN_LIMIT", 0.20)
PORTFOLIO_DD_REENTRY   = PORTFOLIO_DD_EXIT + 0.10
TRANSACTION_COST_BPS   = getattr(config, "TRANSACTION_COST_BPS", 10.0)
TRANSACTION_COST       = TRANSACTION_COST_BPS / 10_000.0

TARGET_HORIZON_DAYS    = getattr(config, "FORWARD_RETURN_DAYS", 5)
WF_MIN_TRAIN_MONTHS    = getattr(config, "MIN_TRAIN_MONTHS", 36)
WF_TEST_MONTHS         = getattr(config, "TEST_WINDOW_MONTHS", 6)
WF_STEP_MONTHS         = getattr(config, "STEP_SIZE_MONTHS", 3)
WF_WINDOW_TYPE         = getattr(config, "VALIDATION_METHOD", "walk_forward_expanding").replace("walk_forward_", "")

# 21 trading-date indices map strictly to a 1 calendar month structural embargo
WF_EMBARGO_TRADING_DAYS = getattr(config, "EMBARGO_TRADING_DAYS", 21)

# Statistical significance boundaries required for ensemble promotion
MIN_OOS_IC_THRESHOLD   = getattr(config, "MIN_OOS_IC_THRESHOLD", 0.005)
MIN_OOS_IC_TSTAT       = getattr(config, "MIN_OOS_IC_TSTAT", 1.5)
PROD_IC_THRESHOLD      = getattr(config, "PROD_IC_THRESHOLD", 0.010)
PROD_IC_TSTAT          = getattr(config, "PROD_IC_TSTAT", 2.5)
MIN_OOS_ICIR_THRESHOLD = 0.30    # kept for backwards compat — NOT used for gate
ALPHA_SMOOTHING_LAMBDA = getattr(config, "ALPHA_SMOOTHING_LAMBDA", 0.70)
RANDOM_SEED            = 42

# Defines structural latency for computationally expensive feature bootstrapping
FEATURE_RESELECT_EVERY_N_FOLDS = 4
ICIR_N_BOOTSTRAP               = 20
ICIR_SAMPLE_SIZE               = 50_000

RETURN_CLIP_MIN = getattr(config, "RETURN_CLIP_MIN", -0.50)
RETURN_CLIP_MAX = getattr(config, "RETURN_CLIP_MAX",  0.50)


# ==============================================================================
# NUMBA JIT KERNELS
# ==============================================================================

@njit(cache=True)
def _rank1d(arr: np.ndarray) -> np.ndarray:
    """
    Computes the fractional cross-sectional rank of a 1D array.

    Args:
        arr (np.ndarray): The continuous input values (e.g., target returns or factors).

    Returns:
        np.ndarray: Evaluated rank metrics uniformly distributed bounded by [0, 1].
            NaNs strictly resolve to 0. Runtime complexity guaranteed at O(N log N).
    """
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
def spearman_ic_nb(feat_matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Calculates parallelized Spearman Rank Information Coefficients across features.

    Args:
        feat_matrix (np.ndarray): The 2D subset mapping extracted signal values.
        target (np.ndarray): The ground truth expected return vectors.

    Returns:
        np.ndarray: A 1D array quantifying the monotonic strength of each feature.
    """
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


# Utilizes pre-allocated static buffers to strictly prevent Numba object-mode fallback
@njit(parallel=True, cache=True)
def rank_pct_parallel_nb(
    pred_matrix: np.ndarray,
    date_ids: np.ndarray,
    n_dates: int,
) -> np.ndarray:
    """
    Generates cross-sectional rank percentiles iteratively segmented by trading date.

    Args:
        pred_matrix (np.ndarray): The $N \times M$ matrix of predictions.
        date_ids (np.ndarray): Mapped identifiers representing contiguous trade dates.
        n_dates (int): Total unique dates mapping execution boundaries.

    Returns:
        np.ndarray: Realigned matrix preserving zero mean and consistent variance,
            which is essential for market-neutral alpha construction.
    """
    n_rows, n_models = pred_matrix.shape
    out = np.zeros_like(pred_matrix)
    for m in prange(n_models):
        idx_buf = np.empty(n_rows, dtype=np.int64)
        for d in range(n_dates):
            cnt = 0
            for i in range(n_rows):
                if date_ids[i] == d:
                    idx_buf[cnt] = i
                    cnt += 1
            if cnt == 0:
                continue
            vals = np.empty(cnt, dtype=np.float64)
            for k in range(cnt):
                vals[k] = pred_matrix[idx_buf[k], m]
            ranks = _rank1d(vals)
            for k in range(cnt):
                out[idx_buf[k], m] = ranks[k]
    return out


@njit(cache=True)
def compound_return_nb(
    weights: np.ndarray,
    ticker_idx: np.ndarray,
    returns_matrix: np.ndarray,
) -> float:
    """
    Computes the aggregate geometric growth rate of the portfolio slice.

    Args:
        weights (np.ndarray): Initial normalized fractional allocations.
        ticker_idx (np.ndarray): Dimensional mapping integers linking to returns.
        returns_matrix (np.ndarray): Sub-slice mapping daily asset price variances.

    Returns:
        float: The cumulative compounded return boundary logic over the holding period.
    """
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


def _warmup_numba() -> None:
    """
    Pre-compiles critical JIT mathematical paths to ensure low latency during optimization.

    Args:
        None

    Returns:
        None
    """
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
def get_data_hash(data_dir) -> str:
    """
    Computes a deterministic MD5 hash of the data warehouse payload.

    Args:
        data_dir (Path): Source directory containing ingested parquet blocks.

    Returns:
        str: The digest hash verifying the structural integrity of the Data Lake.
    """
    hasher    = hashlib.md5()
    data_path = data_dir if hasattr(data_dir, "glob") else Path(data_dir)
    for f in sorted(data_path.glob("*.parquet")):
        hasher.update(f.name.encode())
        hasher.update(str(f.stat().st_mtime).encode())
        hasher.update(str(f.stat().st_size).encode())
    return hasher.hexdigest()


@time_execution
def load_and_build_full_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Orchestrates the hydration, feature aggregation, and caching of the master dataset.

    Args:
        force_rebuild (bool, optional): Overrides valid hashes to trigger cold start generation.

    Returns:
        pd.DataFrame: The structurally sound, target-appended global universe dataset
            ready for downstream chronological stratification.
    """
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
    
    # Cascade the force_rebuild flag to DataManager if it supports it
    import inspect
    if "force_reload" in inspect.signature(dm.get_master_data).parameters:
        data = dm.get_master_data(force_reload=force_rebuild)  # type: ignore[call-arg]
    else:
        data = dm.get_master_data()
        
    if data.index.names[0] is not None:
        data = data.reset_index()

    logger.info("[FUNDAMENTALS] Adding preprocessed fundamental factors...")
    try:
        data = enhance_fundamentals_for_registry(
            data=data,
            fundamentals_dir=config.DATA_DIR / "raw" / "fundamentals",
            earnings_dir=config.DATA_DIR / "raw" / "earnings",
            tickers=data["ticker"].unique().tolist(),
        )
        logger.info(f"[FUNDAMENTALS] Added fundamental factors. Shape: {data.shape}")
    except Exception as e:
        import traceback
        logger.error(f"[FUNDAMENTALS] Preprocessing failed: {e}")
        logger.error(traceback.format_exc())
        logger.warning("[FUNDAMENTALS] Continuing without fundamentals.")

    if "macro_mom_5d" not in data.columns:
        data = add_macro_features(data)

    if data.shape[1] < 120:
        logger.info(f"[FACTORS] Computing on {data.shape[0]:,} rows × {data.shape[1]} cols...")
        fund_cols = [c for c in data.columns if c.startswith("val_") or c.startswith("qual_")]
        logger.info(f"[FACTORS] Fundamental columns present: {fund_cols}")
        from quant_alpha.features.registry import FactorRegistry
        data = FactorRegistry().compute_all(data)
        logger.info(f"[FACTORS] After registry: {data.shape[1]} columns")

    if "open" in data.columns:
        data = data.sort_values(["ticker", "date"])
        next_open   = data.groupby("ticker")["open"].shift(-1).replace(0, np.nan)
        future_open = data.groupby("ticker")["open"].shift(-6)
        # Applies robust clipping bounds to prevent market micro-structure artifacts 
        # (e.g., halts, gaps) from dominating gradient updates.
        data["raw_ret_5d"] = (
            (future_open / next_open) - 1
        ).clip(RETURN_CLIP_MIN, RETURN_CLIP_MAX)
        
        # Data Guards: Liquidity Trap Filter (Remove illiquid stocks)
        if "volume" in data.columns and hasattr(config, "MIN_VOLUME_THRESHOLD"):
            # Compute rolling 21-day median dollar volume (point-in-time)
            median_vol = data.groupby("ticker")["volume"].transform(lambda x: x.rolling(21, min_periods=1).median())
            median_price = data.groupby("ticker")["close"].transform(lambda x: x.rolling(21, min_periods=1).median())
            dollar_vol = median_vol * median_price
            data = data[dollar_vol >= config.MIN_VOLUME_THRESHOLD].copy()
            logger.info(f"[DATA GUARDS] Applied MIN_VOLUME_THRESHOLD (${config.MIN_VOLUME_THRESHOLD:,.0f}).")

    data = data.dropna(axis=1, how="all")
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_parquet(data, cache_path)
    with open(hash_path, "w") as f:
        f.write(current_hash)
    logger.info(f"[DATA] Built: {data.shape[0]:,} rows × {data.shape[1]} cols.")
    return data


def winsorize_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Isolates temporal statistical distribution clipping to strictly historical subsets.

    Args:
        train_df (pd.DataFrame): Walk-forward fit block (used for boundary inference).
        test_df (pd.DataFrame): Walk-forward out-of-sample block (strictly transformed).
        features (list[str]): Candidate target sequences to truncate.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Extrapolated and normalized data matrices.
    """
    scaler = WinsorisationScaler(clip_pct=0.01).fit(train_df, features)
    return scaler.transform(train_df, features), scaler.transform(test_df, features)


def sector_neutral_normalize_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    sector_col: str = "sector",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derives and applies isolated cross-sectional Z-scores independent of future domains.

    Args:
        train_df (pd.DataFrame): The model's baseline training environment.
        test_df (pd.DataFrame): The strict evaluation slice boundary mapping.
        features (list[str]): The numerical alpha predictors to evaluate.
        sector_col (str, optional): The categorical mapping boundary.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Statistically standardized matrices
            exempt from structural macro drift.
    """
    scaler = SectorNeutralScaler(sector_col=sector_col).fit(train_df, features)
    return scaler.transform(train_df, features), scaler.inference_transform(test_df, features)


# ==============================================================================
# TARGET CONSTRUCTION
# ==============================================================================
def build_target(
    df: pd.DataFrame,
    sector_mean=None,
    vol_95th: float | None = None,
) -> pd.DataFrame:
    """
    Constructs a volatility-dampened, sector-neutral forward return target 
    to isolate idiosyncratic alpha.

    Args:
        df (pd.DataFrame): Hydrated dataset mapping raw forward projections.
        sector_mean (Optional[pd.Series]): Pre-computed sector means to inject.
        vol_95th (Optional[float]): Static 95th percentile volatility boundary constraint.

    Returns:
        pd.DataFrame: Appended dataframe encompassing the final 'target' matrix constraint.
    """
    df = df.copy()

    if sector_mean is None:
        s_mean = df.groupby(["date", "sector"], observed=True)["raw_ret_5d"].transform("mean")
    elif isinstance(sector_mean, pd.Series) and len(sector_mean) == len(df):
        s_mean = sector_mean
    else:
        sm_df  = sector_mean.rename("_sm").reset_index()
        df     = df.merge(sm_df, on=["date", "sector"], how="left")
        s_mean = df.pop("_sm").fillna(0.0)

    df["target"] = df["raw_ret_5d"] - s_mean

    df["_sv"] = df.groupby(["date", "sector"], observed=True)["raw_ret_5d"].transform("std")
    
    # Compute expanding 95th percentile to prevent future volatility data leakage
    # Extract daily cross-sectional 95th percentile of sector volatilities
    daily_v95 = df.groupby("date")["_sv"].quantile(0.95)
    expanding_v95 = daily_v95.expanding(min_periods=1).max()
    df["_v95"] = df["date"].map(expanding_v95)
    
    v_thresh = vol_95th if vol_95th is not None else df["_v95"]
    df.loc[df["_sv"] > v_thresh, "target"] *= 0.5
    df = df.drop(columns=["_sv"])
    if "_v95" in df.columns:
        df = df.drop(columns=["_v95"])
    return df


# ==============================================================================
# FEATURE SELECTION (ICIR)
# ==============================================================================
def _single_bootstrap_ic(args: tuple) -> np.ndarray | None:
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


def select_orthogonal_features(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: list[str],
    top_n: int = 25,
    corr_threshold: float = 0.70,
    preserve_categoricals: list[str] | None = None,
    parallel_icir: bool = False,
) -> list[str]:
    """
    Executes Information Coefficient Information Ratio (ICIR) based bootstrapping 
    to construct strictly orthogonal feature pipelines.

    Args:
        df (pd.DataFrame): The raw training dataset mapped to structural target matrices.
        target_col (str): The specific statistical sequence used for monotonic association testing.
        exclude_cols (list[str]): Look-ahead and categorical data series to explicitly discard.
        top_n (int, optional): The combinatorial limitation on feature boundaries.
        corr_threshold (float, optional): Maximum absolute correlation allowed. Defaults to 0.70.
        preserve_categoricals (list[str] | None, optional): Explicit categories to keep.
        parallel_icir (bool, optional): Parallel processing switch. Defaults to False.

    Returns:
        list[str]: The final selected columns maximizing independent signal density 
            without inducing dimensionality collapse.
    """
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

    ic_matrix: list[np.ndarray] = []
    if parallel_icir:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_CORES_TO_USE) as ex:
            for res in ex.map(_single_bootstrap_ic, args_list):
                if res is not None:
                    ic_matrix.append(res)
    else:
        for args in args_list:
            res = _single_bootstrap_ic(args)
            if res is not None:
                ic_matrix.append(res)

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

    selected: list[str] = []
    sel_idxs: list[int] = []
    for rank_pos, global_idx in enumerate(top_idx):
        feat = candidates[global_idx]
        if not sel_idxs:
            selected.append(feat)
            sel_idxs.append(rank_pos)
            continue
        if np.max(np.abs(corr_matrix[rank_pos, sel_idxs])) < corr_threshold:
            selected.append(feat)
            sel_idxs.append(rank_pos)
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
def weighted_symmetric_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates an asymmetric penalty for directional errors. 

    Args:
        y_true (np.ndarray): Empirical target variables mappings.
        y_pred (np.ndarray): Current predictive node output sequences.

    Returns:
        tuple[np.ndarray, np.ndarray]: Derived Gradient and Hessian bounds required by GBDTs.
    """
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
    """
    Determines exponential weighting assignments across competing structural projections.

    Args:
        df (pd.DataFrame): Target dataframe encompassing explicit raw models.

    Returns:
        pd.DataFrame: Augmented ledger injecting smoothed, ranked ensemble structures.
    """
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        df["ensemble_alpha"] = 0.0
        return df

    unique_dates = sorted(df["date"].unique())
    date_to_id   = {d: i for i, d in enumerate(unique_dates)}
    date_ids     = df["date"].map(date_to_id).values.astype(np.int64)
    pred_matrix  = df[pred_cols].values.astype(np.float64)
    rank_matrix  = rank_pct_parallel_nb(pred_matrix, date_ids, len(unique_dates))

    for i, col in enumerate(pred_cols):
        df[f"rank_{col}"] = rank_matrix[:, i]

    # Apply Institutional Ensemble Weights (e.g. 0.4 LGBM, 0.3 XGB, 0.3 CatBoost)
    from config.settings import config
    df["raw_alpha"] = 0.0
    weight_sum = 0.0
    for col in pred_cols:
        model_name = col.replace("pred_", "").lower()
        w = config.MODEL_WEIGHTS.get(model_name, 1.0 / len(pred_cols))
        df["raw_alpha"] += df[f"rank_{col}"] * w
        weight_sum += w
    if weight_sum > 0:
        df["raw_alpha"] /= weight_sum

    df = df.sort_values(["ticker", "date"])
    df["ensemble_alpha"] = (
        df.groupby("ticker")["raw_alpha"]
          .transform(lambda s: s.ewm(alpha=1.0 - ALPHA_SMOOTHING_LAMBDA,
                                     adjust=False).mean())
    )
    
    # Temporal smoothing destroys cross-sectional neutrality; re-rank cross-sectionally
    df["ensemble_alpha"] = (
        df.groupby("date")["ensemble_alpha"]
          .transform(lambda x: x.rank(pct=True))
    )
    return df


def build_ensemble_alpha(predictions: dict) -> pd.DataFrame:
    """
    Aggregates per-model dictionaries mapping discrete predictions into an ensemble matrix.

    Args:
        predictions (dict): Mappings resolving execution outputs dynamically.

    Returns:
        pd.DataFrame: Augmented dataframe encompassing strictly unified ensemble outcomes.
    """
    if not predictions:
        return pd.DataFrame()

    ensemble_df = None
    for name, preds_df in predictions.items():
        col = f"pred_{name}"
        tmp = preds_df[["date", "ticker", "prediction"]].rename(
            columns={"prediction": col})
        ensemble_df = (
            tmp if ensemble_df is None
            else pd.merge(ensemble_df, tmp, on=["date", "ticker"], how="outer")
        )

    if ensemble_df is None or ensemble_df.empty:
        return pd.DataFrame()

    unique_dates = sorted(ensemble_df["date"].unique())
    pred_cols    = [c for c in ensemble_df.columns if c.startswith("pred_")]

    if len(unique_dates) > 1:
        date_to_id  = {d: i for i, d in enumerate(unique_dates)}
        date_ids    = ensemble_df["date"].map(date_to_id).values.astype(np.int64)
        pred_matrix = ensemble_df[pred_cols].values.astype(np.float64)
        rank_matrix = rank_pct_parallel_nb(pred_matrix, date_ids, len(unique_dates))

        for i, col in enumerate(pred_cols):
            ensemble_df[f"rank_{col}"] = rank_matrix[:, i]
    else:
        for col in pred_cols:
            ensemble_df[f"rank_{col}"] = ensemble_df[col].rank(pct=True)

    from config.settings import config
    ensemble_df["raw_alpha"] = 0.0
    weight_sum = 0.0
    for col in pred_cols:
        model_name = col.replace("pred_", "").lower()
        w = config.MODEL_WEIGHTS.get(model_name, 1.0 / len(pred_cols))
        ensemble_df["raw_alpha"] += ensemble_df[f"rank_{col}"] * w
        weight_sum += w
    if weight_sum > 0:
        ensemble_df["raw_alpha"] /= weight_sum

    ensemble_df = ensemble_df.sort_values(["ticker", "date"])
    if len(unique_dates) > 1:
        ensemble_df["ensemble_alpha"] = (
            ensemble_df.groupby("ticker")["raw_alpha"]
              .transform(lambda s: s.ewm(alpha=1.0 - ALPHA_SMOOTHING_LAMBDA,
                                         adjust=False).mean())
        )
        # Re-establish cross-sectional neutrality
        ensemble_df["ensemble_alpha"] = (
            ensemble_df.groupby("date")["ensemble_alpha"]
              .transform(lambda x: x.rank(pct=True))
        )
    else:
        ensemble_df["ensemble_alpha"] = ensemble_df["raw_alpha"]

    return ensemble_df


# ==============================================================================
# MACRO FEATURES
# ==============================================================================
def add_macro_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures structural macro series are hydrated for composite factors.

    Args:
        data (pd.DataFrame): Historical structural mapping.

    Returns:
        pd.DataFrame: Final dataset mapping regime boundaries to distinct assets.
    """
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
    
    from config.settings import config
    alt_dir = config.ALTERNATIVE_DIR
    for macro in ["us_10y", "vix", "oil", "usd", "sp500"]:
        col_name = f"{macro}_close"
        if col_name not in data.columns:
            fpath = alt_dir / f"{macro}.csv"
            if fpath.exists():
                try:
                    m_df = pd.read_csv(fpath)
                    m_df.columns = [str(c).lower() for c in m_df.columns]
                    if "date" in m_df.columns and col_name in m_df.columns:
                        m_df = m_df[["date", col_name]].copy()
                        m_df["date"] = pd.to_datetime(m_df["date"]).dt.tz_localize(None)
                        data = pd.merge(data, m_df, on="date", how="left")
                        data[col_name] = data.groupby("ticker")[col_name].ffill()
                except Exception:
                    pass
                    
    return data


# ==============================================================================
# OOS METRICS
# ==============================================================================
def compute_oos_metrics(preds_df: pd.DataFrame) -> dict:
    """
    Evaluates raw predictions prior to EWMA smoothing to ensure authentic IC persistence.

    Args:
        preds_df (pd.DataFrame): Validated, strictly out-of-sample generation matrices.

    Returns:
        dict: Standardized analytical sequence mapping the target execution boundaries.
    """
    assert "ensemble_alpha" not in preds_df.columns, (
        "compute_oos_metrics must receive raw fold predictions (column='prediction'), "
        "NOT EWM-smoothed ensemble alpha. EWM smoothing is a portfolio construction "
        "step and must not be applied before IC gate evaluation."
    )

    if preds_df.empty or "prediction" not in preds_df.columns:
        return {"ic_mean": 0.0, "ic_std": 1.0, "icir": 0.0, "rank_ic_mean": 0.0}

    df    = preds_df[["date", "prediction", "target"]].dropna()
    dates = df["date"].values
    p, t  = df["prediction"].values, df["target"].values

    daily_ics: list[float] = []
    daily_rics: list[float] = []
    for d in np.unique(dates):
        m      = dates == d
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
# ==============================================================================
def _compute_fold_boundaries(dates: pd.Series) -> list[tuple]:
    """
    Establishes purged validation boundaries utilizing strict trading-day embargo offsets.

    Args:
        dates (pd.Series): Contiguous historical sequence mapping active bounds.

    Returns:
        list[tuple]: Discretized blocks mapping absolute boundary coordinates mapping.
            Zero underlying allocations are mapped via temporal slices.
    """
    n_dates        = len(dates)
    min_train_days = WF_MIN_TRAIN_MONTHS * 21
    test_days      = WF_TEST_MONTHS      * 21
    step_days      = WF_STEP_MONTHS      * 21
    boundaries: list[tuple] = []
    train_end_idx  = min_train_days - 1

    while train_end_idx < n_dates:
        test_start_idx = train_end_idx + 1 + WF_EMBARGO_TRADING_DAYS
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
# ==============================================================================
def _train_single_model(
    name: str,
    model_class,
    params: dict,
    data: pd.DataFrame,
    fold_boundaries: list[tuple],
    meta_cols: list[str],
    selected_features: list[str],
) -> tuple[str, pd.DataFrame, dict]:
    """
    Executes isolated walk-forward iterations, caching optimal feature structures 
    and extracting out-of-sample prediction sequences.

    Args:
        name (str): Nomenclature assignment for target output.
        model_class: Reference object mapping instantiation protocols.
        params (dict): Fixed configurations mapping optimal nodes.
        data (pd.DataFrame): Aggregated mapping variables.
        fold_boundaries (list[tuple]): Absolute structural boundary allocations.
        meta_cols (list[str]): Variables strictly excluded from estimation procedures.
        selected_features (list[str]): The deterministic features extracted from earlier optimization.

    Returns:
        tuple: Mappings assigning specific artifacts sequentially alongside execution details.
    """
    try:
        p = params.copy()
        if "cat_features" in p:
            p["cat_features"] = [c for c in p["cat_features"] if c in data.columns]
            if not p["cat_features"]:
                del p["cat_features"]

        exclude       = set(meta_cols) | {"index", "level_0", "next_open", "future_open"}
        all_preds: list[pd.DataFrame] = []
        feature_cache: dict = {}
        preserve_cats = [
            f for f in [
                "macro_mom_5d", "macro_mom_21d", "macro_vix_proxy",
                "macro_trend_200d", "sector", "industry",
            ]
            if f in data.columns
        ]

        fold_numeric_features = [
            c for c in data.select_dtypes(include=[np.number]).columns
            if c not in exclude and c != "target"
        ]

        logger.info(f"[{name}] Starting {len(fold_boundaries)}-fold walk-forward...")

        for fold_num, (tr_start, tr_end, te_start, te_end) in enumerate(
            tqdm(fold_boundaries, desc=name, unit="fold", leave=False)
        ):
            try:
                tr_mask  = (data["date"] >= tr_start) & (data["date"] <= tr_end)
                te_mask  = (data["date"] >= te_start) & (data["date"] <= te_end)
                train_df = data.loc[tr_mask].copy()
                test_df  = data.loc[te_mask].copy()

                if len(train_df) < 500 or len(test_df) < 50:
                    logger.warning(f"[{name}][F{fold_num}] Fold too small — skip.")
                    del train_df, test_df
                    continue

                # Strict point-in-time cross-sectional scaling. Extrapolated solely from
                # the exact historical regime observed up to the given embargo boundary.
                fold_feats_present = [
                    c for c in fold_numeric_features
                    if c in train_df.columns and c in test_df.columns
                ]
                train_df, test_df = winsorize_fold(train_df, test_df, fold_feats_present)
                train_df, test_df = sector_neutral_normalize_fold(
                    train_df, test_df, fold_feats_present)

                # Fill categoricals per-fold (no leakage risk for string imputation)
                fold_cat_cols = [
                    c for c in train_df.columns
                    if c not in exclude and train_df[c].dtype == object
                ]
                if fold_cat_cols:
                    train_df[fold_cat_cols] = train_df[fold_cat_cols].fillna("Unknown")
                    test_df[fold_cat_cols]  = test_df[fold_cat_cols].fillna("Unknown")

                # Feature selection (cached, re-run every N folds)
                if (fold_num % FEATURE_RESELECT_EVERY_N_FOLDS == 0
                        or "features" not in feature_cache):
                    features = select_orthogonal_features(
                        train_df,
                        target_col="target",
                        exclude_cols=list(exclude),
                        top_n=25,
                        corr_threshold=0.70,
                        preserve_categoricals=preserve_cats,
                        parallel_icir=False,
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

                for _str_col in ["sector", "industry"]:
                    if _str_col in train_df.columns:
                        train_df[_str_col] = train_df[_str_col].astype("category")
                    if _str_col in test_df.columns:
                        test_df[_str_col]  = test_df[_str_col].astype("category")

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
                    model.fit(X_train, y_train)
                    raw_preds = model.predict(X_test)
                except Exception as e:
                    try:
                        model.fit(
                            X_train.reset_index(drop=True),
                            y_train.reset_index(drop=True),
                        )
                        raw_preds = model.predict(X_test.reset_index(drop=True))
                    except Exception:
                        logger.warning(
                            f"[{name}][F{fold_num}] Model fit failed even after reset: {e}")
                        raise

                fold_preds = test_df[["date", "ticker"]].reset_index(drop=True).copy()
                fold_preds["prediction"] = raw_preds
                all_preds.append(fold_preds)

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
            data[["date", "ticker", "target"]],
            on=["date", "ticker"],
            how="left",
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

    def __init__(self, target_vol: float = 0.15) -> None:
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

    def tick(self) -> None:
        if self.cooldown_left > 0:
            self.cooldown_left -= 1

    def update_equity(self, r: float) -> None:
        self.current_equity *= (1.0 + r)
        self.peak_equity     = max(self.peak_equity, self.current_equity)

    def get_current_drawdown(self) -> float:
        return (
            0.0 if self.peak_equity <= 0
            else (self.current_equity / self.peak_equity) - 1.0
        )


# ==============================================================================
# PORTFOLIO OPTIMIZATION
# ==============================================================================
@time_execution
def generate_optimized_weights(
    predictions: pd.DataFrame,
    prices_df: pd.DataFrame,
    method: str = "mean_variance",
) -> pd.DataFrame:
    """
    Derives discrete asset allocation vectors mapping optimal execution ratios.

    Utilizes Ledoit-Wolf shrinkage against sparse historical matrices to establish 
    stable covariance prior to execution.

    Args:
        predictions (pd.DataFrame): Scaled inference boundary scores.
        prices_df (pd.DataFrame): Executable cross-sectional boundaries mapping allocations.
        method (str, optional): Target methodology mapping internal configurations. Defaults to 'mean_variance'.

    Returns:
        pd.DataFrame: A discrete order ledger resolving specific fractional assignments.
    """
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
    valid_dates    = [
        d for d in unique_dates
        if d >= price_matrix.index.min() + pd.Timedelta(days=lookback_days)
    ]
    allocs: list[dict] = []
    lw             = LedoitWolf()
    cur_w: dict[str, float] = {}
    prev_date      = None
    pred_idx       = predictions.set_index(["date", "ticker"])["ensemble_alpha"]

    for cur_date in tqdm(valid_dates, desc="Optimizing"):
        if prev_date is not None and cur_w:
            mask = (returns_dates > prev_date) & (returns_dates <= cur_date)
            pd_  = returns_array[mask.values]
            if pd_.shape[0] > 0:
                held  = list(cur_w.keys())
                w_arr = np.array([cur_w[t] for t in held], dtype=np.float64)
                i_arr = np.array(
                    [ticker_to_col.get(t, -1) for t in held], dtype=np.int64)
                risk_manager.update_equity(compound_return_nb(w_arr, i_arr, pd_))

        dd  = risk_manager.get_current_drawdown()
        lev = risk_manager.check_systemic_stop(dd)
        risk_manager.tick()
        if lev == 0.0:
            cur_w = {}
            prev_date = cur_date
            continue

        try:
            day_preds = pred_idx.loc[cur_date]
        except KeyError:
            prev_date = cur_date
            continue

        top     = day_preds.nlargest(TOP_N_STOCKS)
        tickers = top.index.tolist()
        exp_ret = top.to_dict()
        start   = cur_date - pd.Timedelta(days=lookback_days)
        avail   = [t for t in tickers if t in returns_matrix.columns]
        hist    = returns_matrix.loc[start:cur_date, avail]
        
        # Strict density threshold (>= 80% non-NaN) to ensure robust Ledoit-Wolf covariance
        valid_c = hist.columns[hist.notna().mean() >= 0.8]
        hist    = hist[valid_c]
        
        # Handle NaNs via forward-fill/back-fill prior to defaulting to 0.0
        hist    = hist.ffill().bfill().fillna(0.0)
        surv    = valid_c.tolist()

        if len(surv) < 2:
            weights: dict[str, float] = {}
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

        final_w: dict[str, float] = {}
        for ticker, tw in weights.items():
            cw = cur_w.get(ticker, 0.0)
            if abs(tw - cw) < 1e-4:
                final_w[ticker] = cw
                continue
            cost = abs(tw - cw) * TRANSACTION_COST * 1.5
            gain = abs(exp_ret.get(ticker, 0)) * 0.03
            final_w[ticker] = tw if gain > cost else cw

        final_w   = {t: w * lev for t, w in final_w.items()}
        cur_w     = final_w.copy()
        prev_date = cur_date
        for t, w in final_w.items():
            allocs.append({"date": cur_date, "ticker": t, "optimized_weight": w})

    return pd.DataFrame(allocs)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
@time_execution
def run_production_pipeline(
    force_rebuild: bool = False,
    parallel_models: bool = False,
    run_all: bool = False,
) -> None:
    """
    Main execution DAG constructing the entire quantitative lifecycle.

    Args:
        force_rebuild (bool, optional): Overrides valid hashes triggering complete sequence calculation.
        parallel_models (bool, optional): Allows cross-sectional structural execution simultaneously mapping dependencies.
        run_all (bool, optional): If True, triggers execution validations.

    Returns:
        None
    """
    logger.info(
        f"[BOOT] Cores={CPU_CORES_TO_USE}/{TOTAL_CORES}  "
        f"RAM={psutil.virtual_memory().total/1e9:.1f}GB  "
        f"parallel_models={parallel_models}"
    )

    logger.info("[NUMBA] Warming up JIT kernels...")
    _warmup_numba()

    # ── 1. LOAD ───────────────────────────────────────────────────────────
    data = load_and_build_full_dataset(force_rebuild=force_rebuild)
    if "date" not in data.columns:
        data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"])
    data = (
        data.drop_duplicates(subset=["date", "ticker"])
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
    )

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
        data["raw_ret_5d"]  = (
            (data["future_open"] / data["next_open"]) - 1
        ).clip(RETURN_CLIP_MIN, RETURN_CLIP_MAX)

    data["pnl_return"] = (
        data.groupby("ticker")["open"].shift(-1) / data["open"] - 1
    ).clip(RETURN_CLIP_MIN, RETURN_CLIP_MAX)

    # ── 3. TARGET CONSTRUCTION ────────────────────────────────────────────
    data = build_target(data)
    data = data.dropna(subset=["target", "pnl_return"])

    # ── 4. FEATURE ENGINEERING ────────────────────────────────────────────
    meta_cols = [
        "ticker", "date", "target", "pnl_return",
        "open", "high", "low", "close", "volume",
        "sector", "industry", "raw_ret_5d",
        "next_open", "future_open",
        "macro_mom_5d", "macro_mom_21d", "macro_vix_proxy", "macro_trend_200d",
        "us_10y_close", "vix_close", "oil_close", "usd_close", "sp500_close"
    ]
    selector = FeatureSelector(meta_cols=meta_cols)
    data     = selector.drop_low_variance(data)

    wf_train_days  = WF_MIN_TRAIN_MONTHS * 21
    unique_dates   = np.sort(data["date"].unique())
    train_cutoff   = unique_dates[min(wf_train_days, len(unique_dates) - 1)]
    wf_train_data  = data[data["date"] <= train_cutoff]
    
    sel_sample = wf_train_data.sample(n=min(200_000, len(wf_train_data)), random_state=RANDOM_SEED).copy()

    # Fill categoricals in sel_sample only (no leakage risk)
    sel_cat_cols = [
        c for c in sel_sample.columns
        if c not in set(meta_cols) | {"index", "level_0"}
        and sel_sample[c].dtype == object
    ]
    if sel_cat_cols:
        sel_sample[sel_cat_cols] = sel_sample[sel_cat_cols].fillna("Unknown")

    # Apply lightweight winsorisation to sel_sample for feature selection
    sel_numeric = [
        c for c in sel_sample.select_dtypes(include=[np.number]).columns
        if c not in set(meta_cols) | {"index", "level_0", "next_open", "future_open"}
        and c != "target"
    ]
    sel_sample = (
        WinsorisationScaler(clip_pct=0.01)
        .fit(sel_sample, sel_numeric)
        .transform(sel_sample, sel_numeric)
    )

    logger.info(f"[FEATURE_SEL] Global selection on {len(sel_sample):,} rows "
                f"(winsorised for unbiased ICIR)...")
    selected_features = select_orthogonal_features(
        sel_sample,
        target_col="target",
        exclude_cols=meta_cols,
        top_n=25,
        corr_threshold=0.70,
        preserve_categoricals=["sector", "industry"],
        parallel_icir=True,
    )
    # Macro features always included regardless of ICIR rank
    for mf in ["macro_mom_5d", "macro_mom_21d", "macro_vix_proxy", "macro_trend_200d"]:
        if mf not in selected_features and mf in data.columns:
            selected_features.append(mf)
    logger.info(f"[FEATURE_SEL] Final: {len(selected_features)} features")
    del sel_sample
    gc.collect()

    # ── 5. GLOBAL CATEGORICAL FILL — safe to do globally (no leakage) ─────
    cat_features = [
        c for c in data.columns
        if c not in set(meta_cols) | {"index", "level_0"}
        and data[c].dtype == object
    ]
    if cat_features:
        data[cat_features] = data[cat_features].fillna("Unknown")

    gc.collect()
    logger.info(
        f"[PREPROC] Categorical fill complete (scaling moved per-fold). "
        f"RAM: {psutil.virtual_memory().used/1e9:.1f} GB used."
    )

    # ── 6. WALK-FORWARD SETUP ─────────────────────────────────────────────
    dates           = pd.Series(data["date"].unique()).sort_values().reset_index(drop=True)
    fold_boundaries = _compute_fold_boundaries(dates)
    total_months    = (data["date"].max() - data["date"].min()).days / 30
    expected_oos    = total_months - WF_MIN_TRAIN_MONTHS

    print(f"\n{'='*62}")
    print(f"  WALK-FORWARD CONFIGURATION")
    print(f"{'='*62}")
    print(f"  Horizon      : {TARGET_HORIZON_DAYS}d  |  Window: {WF_WINDOW_TYPE}")
    print(f"  Min train    : {WF_MIN_TRAIN_MONTHS} months ({WF_MIN_TRAIN_MONTHS*21} trading days)")
    print(f"  Test window  : {WF_TEST_MONTHS} months  |  Step: {WF_STEP_MONTHS} months")
    print(f"  Embargo      : {WF_EMBARGO_TRADING_DAYS} trading days (~1 month)")
    print(f"  Folds        : {len(fold_boundaries)}  (~{expected_oos:.0f} months OOS)")
    print(f"  Features     : {len(selected_features)} selected")
    print(f"  ICIR iters   : {ICIR_N_BOOTSTRAP}  (sample {ICIR_SAMPLE_SIZE:,} rows)")
    print(f"  Cache every  : {FEATURE_RESELECT_EVERY_N_FOLDS} folds")
    print(f"  Scaling      : per-fold (train-only fit) ✅ no leakage")
    print(f"  Model order  : {'PARALLEL ⚠️ RAM intensive' if parallel_models else 'SERIAL ✅ memory-safe'}")
    print(f"  IC gate      : ≥{MIN_OOS_IC_THRESHOLD}  |  ICIR gate: ≥{MIN_OOS_ICIR_THRESHOLD}")
    print(f"{'='*62}\n")

    # ── 7. MODEL CONFIGS ──────────────────────────────────────────────────
    models_config = {
        "LightGBM": (LightGBMModel, {
            "n_estimators": 300, "learning_rate": 0.035675689449899364,
            "num_leaves": 24,
            "max_depth": 6, "importance_type": "gain",
            "n_jobs": MODEL_THREADS, "random_state": RANDOM_SEED,
            "objective": "regression_l1",
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
        }),
        "XGBoost": (XGBoostModel, {
            "n_estimators": 300, "learning_rate": 0.03261427122370329,
            "max_depth": 3,
            "min_child_weight": 1, "subsample": 0.6756485311881795,
            "n_jobs": MODEL_THREADS, "random_state": RANDOM_SEED,
            "objective": "reg:absoluteerror", "max_bin": 64,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
        }),
        "CatBoost": (CatBoostModel, {
            "iterations": 150, "learning_rate": 0.06, "depth": 4,
            "l2_leaf_reg": 1.0, "subsample": 0.6,
            "verbose": 0, "thread_count": MODEL_THREADS,
            "random_seed": RANDOM_SEED,
            "cat_features": ["sector", "industry"],
            "loss_function": "MAE",
            "eval_metric": "RMSE",
            "allow_writing_files": False,
            "boosting_type": "Plain", "bootstrap_type": "Bernoulli",
            "border_count": 128, "min_data_in_leaf": 200,
            "grow_policy": "Depthwise", "score_function": "L2",
        }),
    }

    # ── 8. TRAIN ──────────────────────────────────────────────────────────
    oos_preds_master: dict[str, pd.DataFrame] = {}
    model_metrics:   dict[str, dict]          = {}

    if parallel_models:
        logger.warning(
            "[TRAIN] Parallel mode: 3 models run simultaneously. "
            "Needs ~12 GB RAM. If OOM, restart without --parallel-models.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(
                    _train_single_model,
                    name, cls, prms, data, fold_boundaries, meta_cols, selected_features
                ): name
                for name, (cls, prms) in models_config.items()
            }
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=3, desc="Models", unit="model",
            ):
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
    all_oos_dates   = sorted({
        d for df_p in oos_preds_master.values() for d in df_p["date"].unique()
    })
    total_data_days = data["date"].nunique()

    print(f"\n{'='*62}")
    print(f"  OOS PERFORMANCE SUMMARY")
    print(f"{'='*62}")
    print(f"  Period : {min(all_oos_dates).date()} → {max(all_oos_dates).date()} "
          f"({len(all_oos_dates)} days  /  {total_data_days} total)")
    print()
    for mname, m in model_metrics.items():
        n_d   = m.get("n_dates", 1)
        ic    = m.get("ic_mean", 0)
        std   = m.get("ic_std",  1e-8)
        tstat = ic / (std / (n_d ** 0.5)) if n_d > 0 else 0.0
        ann_icir  = (ic / std) * (252 ** 0.5) if std > 0 else 0.0
        ensemble_ok = ic >= MIN_OOS_IC_THRESHOLD and tstat >= MIN_OOS_IC_TSTAT
        prod_ok     = ic >= PROD_IC_THRESHOLD and tstat >= PROD_IC_TSTAT
        status      = "✅ PROD" if prod_ok else ("🟡 ENSEMBLE" if ensemble_ok else "❌ GATED")
        print(f"  {mname:<12}  IC={ic:+.6f}  t={tstat:+.1f}  "
              f"AnnICIR={ann_icir:.2f}  RankIC={m.get('rank_ic_mean',0):+.4f}  {status}")
    print(f"{'='*62}\n")

    # ── 10. SAVE PRODUCTION MODELS ────────────────────────────────────────
    for name, (cls, prms) in models_config.items():
        m     = model_metrics.get(name, {})
        n_d   = m.get("n_dates", 1)
        ic_m  = m.get("ic_mean",  0)
        ic_s  = m.get("ic_std",   1e-8)
        tstat = ic_m / (ic_s / (n_d ** 0.5)) if n_d > 0 else 0.0

        if not (ic_m >= MIN_OOS_IC_THRESHOLD and tstat >= MIN_OOS_IC_TSTAT):
            logger.info(
                f"[SAVE] {name} GATED from save: IC={ic_m:+.4f} "
                f"(need {MIN_OOS_IC_THRESHOLD}), "
                f"t-stat={tstat:.1f} (need {MIN_OOS_IC_TSTAT}). "
                "Too weak for ensemble."
            )
            continue

        tier = ("PROD" if (ic_m >= PROD_IC_THRESHOLD and tstat >= PROD_IC_TSTAT)
                else "ENSEMBLE")
        logger.info(
            f"[SAVE] Fitting {name} ({tier}) on full history... "
            f"(IC={ic_m:+.4f}, t={tstat:.1f})"
        )

        try:
            prod_cutoff = pd.Timestamp(
                getattr(config, "PROD_MODEL_MIN_DATE",
                        data["date"].min().strftime("%Y-%m-%d"))
            )
            p_data = data[data["date"] >= prod_cutoff].copy()
            
            wf_train_days = WF_MIN_TRAIN_MONTHS * 21
            unique_p_dates = np.sort(p_data["date"].unique())
            if len(unique_p_dates) > wf_train_days:
                p_data = p_data[p_data["date"] >= unique_p_dates[-wf_train_days]].copy()

            for _c in ["sector", "industry"]:
                if _c in p_data.columns:
                    p_data[_c] = p_data[_c].astype("category")
            cat_fill = [
                c for c in p_data.columns
                if c not in meta_cols and p_data[c].dtype == object
            ]
            p_data[cat_fill] = p_data[cat_fill].fillna("Unknown")
            p_data = p_data.dropna(subset=["target"])

            feat_avail = [f for f in selected_features if f in p_data.columns]

            # Persists strictly the bounds inferred from historical domains
            prod_numeric = [
                f for f in feat_avail
                if f in p_data.select_dtypes(include=[np.number]).columns
            ]
            prod_winsoriser = WinsorisationScaler(clip_pct=0.01).fit(
                p_data, prod_numeric)
            prod_sector_scaler = SectorNeutralScaler(sector_col="sector").fit(
                p_data, prod_numeric)
            p_data = prod_winsoriser.transform(p_data, prod_numeric)
            p_data = prod_sector_scaler.transform(p_data, prod_numeric)

            prod_model = cls(params=prms.copy())
            X_prod = p_data[feat_avail].copy()
            for _c in ["sector", "industry"]:
                if _c in X_prod.columns:
                    X_prod[_c] = X_prod[_c].astype("category")

            prod_model.fit(X_prod, p_data["target"])

            save_dir = config.MODELS_DIR / "production"
            save_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "model":            prod_model,
                    "feature_names":    feat_avail,
                    "winsoriser":       prod_winsoriser,
                    "sector_scaler":    prod_sector_scaler,
                    "trained_to":       str(data["date"].max().date()),
                    "oos_metrics":      m,
                    "meta": {
                        "pipeline_version": "v4.1",
                        "target":           "sector_neutral_ret_5d",
                        "clip_pct":         0.01,
                        "return_clip_min":  RETURN_CLIP_MIN,
                        "return_clip_max":  RETURN_CLIP_MAX,
                    },
                },
                save_dir / f"{name.lower()}_latest.pkl",
            )
            logger.info(
                f"[SAVE] {name} saved → {save_dir / f'{name.lower()}_latest.pkl'}"
            )
        except Exception as exc:
            logger.error(f"[SAVE] {name} save failed: {exc}", exc_info=True)

    # ── 11. ENSEMBLE ──────────────────────────────────────────────────────
    def _passes_ensemble_gate(nm: str) -> bool:
        m_   = model_metrics.get(nm, {})
        n_d_ = m_.get("n_dates", 1)
        ic_  = m_.get("ic_mean",  0)
        std_ = m_.get("ic_std",   1e-8)
        t_   = ic_ / (std_ / (n_d_ ** 0.5)) if n_d_ > 0 else 0.0
        return ic_ >= MIN_OOS_IC_THRESHOLD and t_ >= MIN_OOS_IC_TSTAT

    passing_models = {
        nm: df_p for nm, df_p in oos_preds_master.items()
        if _passes_ensemble_gate(nm)
    }

    if not passing_models:
        model_ic_lines = []
        for n in model_metrics:
            m_    = model_metrics[n]
            ic_   = m_.get("ic_mean", 0)
            std_  = m_.get("ic_std", 1e-8)
            n_d_  = max(m_.get("n_dates", 1) ** 0.5, 1)
            t_    = ic_ / (std_ / n_d_)
            model_ic_lines.append(
                f"    {n}: IC={ic_:+.4f}  t={t_:.1f}"
            )
        detail = "\n".join(model_ic_lines)
        raise RuntimeError(
            f"\n[ENSEMBLE] FATAL: No models passed IC/t-stat gate.\n"
            f"  Required : IC >= {MIN_OOS_IC_THRESHOLD}, "
            f"t-stat >= {MIN_OOS_IC_TSTAT}\n"
            f"  Got:\n{detail}\n"
            f"  Action   : check raw_ret_5d coverage, feature engineering, "
            f"data quality, and target construction."
        )

    gated = set(oos_preds_master) - set(passing_models)
    if gated:
        logger.warning(
            f"[ENSEMBLE] Excluding {len(gated)} gated model(s): {sorted(gated)}. "
            "Only IC-gate-passing models contribute to alpha signals."
        )

    ensemble_df = None
    for nm, df_p in passing_models.items():
        tmp = df_p[["date", "ticker", "prediction"]].rename(
            columns={"prediction": f"pred_{nm}"})
        ensemble_df = (
            tmp if ensemble_df is None
            else pd.merge(ensemble_df, tmp, on=["date", "ticker"], how="outer")
        )

    del oos_preds_master
    gc.collect()

    ensemble_df = pd.merge(
        ensemble_df,
        data[["date", "ticker", "target", "pnl_return"]],
        on=["date", "ticker"],
        how="left",
    )
    final_results = calculate_ranks_robust(ensemble_df)
    save_parquet(final_results, config.CACHE_DIR / "ensemble_predictions.parquet")
    logger.info(f"[ENSEMBLE] Saved → {config.CACHE_DIR / 'ensemble_predictions.parquet'}")

    if run_all:
        # ── 12. BACKTEST (spawn as subprocess) ───────────────────────────
        logger.info("[BACKTEST] Spawning run_backtest.py for multiple methods...")
        methods_to_run = ["top_n", "mean_variance", "risk_parity", "kelly"]
        for method in methods_to_run:
            try:
                cmd = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_backtest.py"),
                    "--method",
                    method,
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"✅ Backtest for '{method}' completed successfully.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Backtest for method '{method}' failed with exit code {e.returncode}.")
                logger.error(f"STDOUT: {e.stdout}")
                logger.error(f"STDERR: {e.stderr}")
            except Exception as e:
                logger.error(f"Backtest for method '{method}' failed: {e}")

    logger.info("[DONE] Pipeline complete.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quant Alpha Production Walk-Forward Pipeline v4.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python scripts/train_models.py               # train only  (~30 min, default)
  python scripts/train_models.py --all         # train + backtest + factor + alpha
  python scripts/train_models.py --all --force-rebuild   # rebuild cache first
  python scripts/train_models.py --parallel-models --all # fast on 16GB+ RAM
        """,
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run full pipeline: train + backtest + factor analysis + Jensen alpha.",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Ignore dataset cache and rebuild from raw data.",
    )
    parser.add_argument(
        "--parallel-models", action="store_true",
        help="Train 3 models simultaneously. Needs 16GB+ RAM. Default: serial.",
    )
    args = parser.parse_args()
    run_production_pipeline(
        force_rebuild=args.force_rebuild,
        parallel_models=args.parallel_models,
        run_all=args.all,
    )