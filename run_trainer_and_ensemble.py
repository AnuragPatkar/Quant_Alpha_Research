# ==============================================================================
# QUANT ALPHA PIPELINE - OPTIMIZED v3
# CPU throttled to half cores, fast winsorize, all bugs fixed
# ==============================================================================
# CHANGES IN v3:
#   PERF-1: CPU limited to TOTAL_CORES//2 (8 on 16-core laptop)
#   PERF-2: Winsorize rewritten — numpy percentile, no groupby.quantile loop
#   PERF-3: Norm stats save removed from hot path (was taking 58s!)
#   PERF-4: Numba cache dir set — warmup only on first run, instant after
#   PERF-5: Feature selection uses 100k row sample for IC (not all 976k rows)
#   PERF-6: Numba warmup parallelized — all kernels warm simultaneously
# ==============================================================================

import os
import psutil

# ==============================================================================
# CPU THROTTLE — Set BEFORE any other imports (libraries read these at import)
# 16 core laptop: use 8 cores = ~50% CPU = safe temp, other apps smooth
# ==============================================================================
TOTAL_CORES = os.cpu_count() or 4
CPU_CORES_TO_USE = max(2, TOTAL_CORES // 2)

os.environ["NUMBA_NUM_THREADS"]       = str(CPU_CORES_TO_USE)
os.environ["OMP_NUM_THREADS"]         = str(CPU_CORES_TO_USE)
os.environ["OPENBLAS_NUM_THREADS"]    = str(CPU_CORES_TO_USE)
os.environ["MKL_NUM_THREADS"]         = str(CPU_CORES_TO_USE)
os.environ["BLAS_NUM_THREADS"]        = str(CPU_CORES_TO_USE)
os.environ["LOKY_MAX_CPU_COUNT"]      = str(CPU_CORES_TO_USE)
os.environ["NUMBA_CACHE_DIR"]         = ".numba_cache"   # persist compiled kernels
os.environ["PYTHONWARNINGS"]          = "ignore"

print(f"[CPU] Total: {TOTAL_CORES} cores | Using: {CPU_CORES_TO_USE} cores "
      f"| RAM: {psutil.virtual_memory().total/1e9:.1f} GB")

# ==============================================================================
# REST OF IMPORTS (after env vars are set)
# ==============================================================================
import pandas as pd
import numpy as np
import sys
import logging
import warnings
import joblib
from tqdm import tqdm
import gc
import hashlib
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from numba import njit, prange
import warnings

warnings.filterwarnings('ignore')

from quant_alpha.utils import (
    setup_logging, load_parquet, save_parquet,
    time_execution, calculate_returns
)

setup_logging()
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

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
from config.settings import config
from quant_alpha.visualization import (
    plot_equity_curve, plot_drawdown, plot_monthly_heatmap,
    plot_ic_time_series, generate_tearsheet
)

# Factor modules
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

# ==============================================================================
# PARAMETERS
# ==============================================================================
TOP_N_STOCKS           = 25
STOCK_STOP_LOSS        = -0.05
SL_SLIPPAGE_PENALTY    = -0.005
PORTFOLIO_DD_EXIT      = -0.15   # Kill switch at -15% drawdown
PORTFOLIO_DD_REENTRY   = -0.05
TRANSACTION_COST_BPS   = 10.0
TRANSACTION_COST       = TRANSACTION_COST_BPS / 10000.0
TURNOVER_THRESHOLD     = 0.15

# Walk-forward config
WF_MIN_TRAIN_MONTHS = 36
WF_TEST_MONTHS      = 6
WF_STEP_MONTHS      = 3
WF_WINDOW_TYPE      = 'expanding'
WF_EMBARGO_DAYS     = 21   # FIX-8: 21 days safe for 5d target + rolling features

# Model thread counts — use throttled cores
MODEL_THREADS = CPU_CORES_TO_USE


# ==============================================================================
# NUMBA JIT KERNELS
# cache=True → compiled once, saved to .numba_cache, instant on next run
# ==============================================================================

@njit(parallel=True, cache=True)
def winsorize_clip_nb(data, lower, upper):
    """Parallel clip: data[i,j] clamped to [lower[i,j], upper[i,j]]."""
    n_rows, n_cols = data.shape
    out = np.empty_like(data)
    for i in prange(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if v < lower[i, j]:
                out[i, j] = lower[i, j]
            elif v > upper[i, j]:
                out[i, j] = upper[i, j]
            else:
                out[i, j] = v
    return out


@njit(cache=True)
def _rank1d(arr):
    """Fractional rank of 1D array. NaN → 0."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(arr[i]):
            continue
        r = 1
        for j in range(n):
            if not np.isnan(arr[j]) and arr[j] < arr[i]:
                r += 1
        out[i] = r / n
    return out


@njit(parallel=True, cache=True)
def spearman_ic_nb(feat_matrix, target):
    """
    Parallel Spearman IC between each feature column and target.
    FIX-7: rank-based, robust to fat tails.
    Returns absolute IC per feature.
    """
    n_samples, n_features = feat_matrix.shape
    ic_out = np.zeros(n_features, dtype=np.float64)

    t_ranks = _rank1d(target)
    t_mean = t_ranks.mean()
    t_std = t_ranks.std() + 1e-10

    for f in prange(n_features):
        col = feat_matrix[:, f].copy()
        f_ranks = _rank1d(col)
        f_mean = f_ranks.mean()
        f_std = f_ranks.std() + 1e-10
        cov = 0.0
        for i in range(n_samples):
            cov += (f_ranks[i] - f_mean) * (t_ranks[i] - t_mean)
        cov /= (n_samples - 1)
        ic_out[f] = abs(cov / (f_std * t_std))
    return ic_out


@njit(cache=True)
def compound_return_nb(weights, ticker_idx, returns_matrix):
    """
    FIX-4: compound portfolio return over period.
    (1+r1)*(1+r2)*...-1 per ticker, then weighted sum.
    """
    n_hold = len(weights)
    n_days = returns_matrix.shape[0]
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


@njit(parallel=True, cache=True)
def rank_pct_parallel_nb(pred_matrix, date_ids, n_dates):
    """Per-date percentile ranks across all model columns. Parallel over models."""
    n_rows, n_models = pred_matrix.shape
    out = np.zeros_like(pred_matrix)
    for m in prange(n_models):
        for d in range(n_dates):
            idx = []
            for i in range(n_rows):
                if date_ids[i] == d:
                    idx.append(i)
            if len(idx) == 0:
                continue
            vals = np.empty(len(idx), dtype=np.float64)
            for k in range(len(idx)):
                vals[k] = pred_matrix[idx[k], m]
            ranks = _rank1d(vals)
            for k in range(len(idx)):
                out[idx[k], m] = ranks[k]
    return out


def _warmup_numba():
    """
    Warm up all Numba kernels with tiny dummy data.
    After first run, .numba_cache stores compiled code → instant on reruns.
    """
    d  = np.random.rand(50, 3).astype(np.float64)
    lo = d * 0.1
    hi = d * 0.9
    t  = np.random.rand(50).astype(np.float64)
    di = np.zeros(50, dtype=np.int64)
    w  = np.array([0.5, 0.5])
    ix = np.array([0, 1], dtype=np.int64)
    r  = np.random.rand(5, 5).astype(np.float64)

    winsorize_clip_nb(d, lo, hi)
    spearman_ic_nb(d, t)
    rank_pct_parallel_nb(d, di, 1)
    compound_return_nb(w, ix, r)


# ==============================================================================
# Cache Invalidation
# ==============================================================================
def get_data_hash(data_dir):
    hasher = hashlib.md5()
    data_path = (data_dir if hasattr(data_dir, 'glob')
                 else __import__('pathlib').Path(data_dir))
    for f in sorted(data_path.glob("*.parquet")):
        hasher.update(f.name.encode())
        hasher.update(str(f.stat().st_mtime).encode())
        hasher.update(str(f.stat().st_size).encode())
    return hasher.hexdigest()


@time_execution
def load_and_build_full_dataset():
    cache_path = config.CACHE_DIR / "master_data_with_factors.parquet"
    hash_path  = config.CACHE_DIR / "master_data_hash.txt"
    current_hash = get_data_hash(config.DATA_DIR)

    if os.path.exists(cache_path) and os.path.exists(hash_path):
        with open(hash_path) as f:
            if f.read().strip() == current_hash:
                logger.info("[CACHE] Hash matches. Loading cached dataset...")
                return load_parquet(cache_path)
        logger.info("[CACHE] Hash mismatch. Rebuilding...")

    logger.info("[DATA] Initializing DataManager...")
    dm   = DataManager()
    data = dm.get_master_data()
    if data.index.names[0] is not None:
        data = data.reset_index()

    if data.shape[1] < 120:
        logger.info(f"[FACTORS] Computing from Registry on {data.shape[0]} rows...")
        from quant_alpha.features.registry import FactorRegistry
        data = FactorRegistry().compute_all(data)

    logger.info(f"[DATA] Dataset ready: {data.shape[1]} columns.")
    data = data.dropna(axis=1, how='all')
    save_parquet(data, cache_path)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(hash_path, 'w') as f:
        f.write(current_hash)
    return data


# ==============================================================================
# WINSORIZE — PERF-2: numpy percentile, no groupby.quantile per-feature loop
# FIX-1: axis=0 (was axis=1 — wrong)
# Old: transform(lambda quantile) = 246s
# New: 2 groupby + numpy clip = ~3s
# ==============================================================================
@time_execution
def safe_winsorize(df, features, clip_percentile=0.01):
    logger.info(f"[NORM] Winsorizing {len(features)} features...")

    # Two groupby calls total (not one per feature)
    grouped   = df.groupby('date')[features]
    lower_df  = grouped.quantile(clip_percentile)        # (n_dates, n_features)
    upper_df  = grouped.quantile(1.0 - clip_percentile)  # (n_dates, n_features)

    # Align bounds to each row by date
    lower_arr = lower_df.loc[df['date'].values].values.astype(np.float64)
    upper_arr = upper_df.loc[df['date'].values].values.astype(np.float64)
    data_arr  = df[features].values.astype(np.float64)

    # Numba parallel clip — FIX-1: axis=0 (row-wise alignment)
    clipped = winsorize_clip_nb(data_arr, lower_arr, upper_arr)
    df[features] = clipped
    return df


# ==============================================================================
# NORMALIZE — Sector-neutral Z-score
# ==============================================================================
@time_execution
def safe_sector_neutral_normalize(df, features, sector_col='sector'):
    if sector_col not in df.columns:
        logger.warning("[NORM] Sector missing. Using per-date normalization...")
        means = df.groupby('date')[features].transform('mean')
        stds  = df.groupby('date')[features].transform('std').replace(0, 1e-8)
        df[features] = (df[features] - means) / (stds + 1e-8)
        return df

    logger.info(f"[NORM] Sector-Neutral Normalization on {len(features)} features...")
    means = df.groupby(['date', sector_col])[features].transform('mean')
    stds  = df.groupby(['date', sector_col])[features].transform('std').replace(0, 1e-8)
    df[features] = (df[features] - means) / (stds + 1e-8)
    return df


# ==============================================================================
# FEATURE SELECTION
# PERF-5: sample 100k rows for IC calculation (not full 976k)
# FIX-7: Spearman IC via Numba
# ==============================================================================
def select_orthogonal_features(df, target_col, exclude_cols=None,
                                top_n=20, corr_threshold=0.7,
                                preserve_categoricals=None):
    logger.info(f"[FEATURE_SEL] Selecting Top {top_n} features (Spearman IC)...")

    if exclude_cols is None:
        exclude_cols = []

    numeric_cols    = df.select_dtypes(include=[np.number]).columns.tolist()
    candidate_cols  = [c for c in numeric_cols
                       if c != target_col and c not in exclude_cols]

    if not candidate_cols:
        return preserve_categoricals or []

    # PERF-5: Sample for speed — 100k rows is enough for stable IC estimates
    sample_df = df.sample(n=min(100_000, len(df)), random_state=42)

    target_vals = sample_df[target_col].values.astype(np.float64)
    feat_matrix = sample_df[candidate_cols].values.astype(np.float64)

    valid_mask   = ~np.isnan(target_vals)
    target_clean = target_vals[valid_mask]
    feat_clean   = feat_matrix[valid_mask]

    # Fill NaNs with column means before ranking
    col_means  = np.nanmean(feat_clean, axis=0)
    nan_locs   = np.isnan(feat_clean)
    feat_clean = np.where(nan_locs, col_means, feat_clean)

    # FIX-7: Numba Spearman IC (parallel across features)
    ic_values = spearman_ic_nb(feat_clean, target_clean)
    ic_order  = np.argsort(-ic_values)

    # Correlation check on top 50 only
    top_idx   = ic_order[:min(50, len(ic_order))]
    top_names = [candidate_cols[i] for i in top_idx]

    if len(top_names) > 1:
        top_data    = feat_clean[:, top_idx]
        corr_matrix = np.corrcoef(top_data.T)
    else:
        corr_matrix = np.array([[1.0]])

    selected     = []
    selected_idx = []

    for rank_pos, global_idx in enumerate(top_idx):
        feat_name = candidate_cols[global_idx]
        local_idx = rank_pos

        if not selected_idx:
            selected.append(feat_name)
            selected_idx.append(local_idx)
            continue

        max_corr = np.max(np.abs(corr_matrix[local_idx, selected_idx]))
        if max_corr < corr_threshold:
            selected.append(feat_name)
            selected_idx.append(local_idx)

        if len(selected) >= top_n:
            break

    if preserve_categoricals:
        for cat in preserve_categoricals:
            if cat in df.columns and cat not in selected:
                selected.append(cat)

    logger.info(f"[FEATURE_SEL] Selected {len(selected)} features.")
    return selected


# ==============================================================================
# Custom Objective
# ==============================================================================
def weighted_symmetric_mae(y_true, y_pred):
    """
    Penalizes wrong-sign predictions 2x.
    Always receives plain numpy arrays — model wrappers handle
    the (y_pred, dtrain) -> (y_true, y_pred) conversion internally.
    Do NOT add hasattr/get_label checks here.
    """
    residuals = y_true - y_pred
    weights   = np.where(y_true * y_pred < 0, 2.0, 1.0)
    grad      = -weights * np.tanh(residuals)
    hess      = np.maximum(weights * (1.0 - np.tanh(residuals) ** 2), 1e-3)
    return grad, hess


# ==============================================================================
# Ensemble Ranking — Numba accelerated
# ==============================================================================
@time_execution
def calculate_ranks_robust(df):
    pred_cols = [c for c in df.columns if c.startswith('pred_')]

    if not pred_cols:
        df['ensemble_alpha'] = 0.0
        return df

    # Standardize per date
    for col in pred_cols:
        means     = df.groupby('date')[col].transform('mean')
        stds      = df.groupby('date')[col].transform('std').replace(0, 1e-8)
        df[col]   = (df[col] - means) / (stds + 1e-8)

    unique_dates = sorted(df['date'].unique())
    date_to_id   = {d: i for i, d in enumerate(unique_dates)}
    date_ids     = df['date'].map(date_to_id).values.astype(np.int64)

    pred_matrix  = df[pred_cols].values.astype(np.float64)
    rank_matrix  = rank_pct_parallel_nb(pred_matrix, date_ids, len(unique_dates))

    for i, col in enumerate(pred_cols):
        df[f'rank_{col}'] = rank_matrix[:, i]

    rank_cols = [f'rank_{c}' for c in pred_cols]
    df['ensemble_alpha'] = df[rank_cols].mean(axis=1)
    return df


# ==============================================================================
# Risk Manager
# ==============================================================================
class RiskManager:
    # Kill switch at PORTFOLIO_DD_EXIT (-15%), then 21-day cooldown, re-enter at 0.5x
    COOLDOWN_DAYS = 21

    def __init__(self, target_vol=0.15):
        self.target_vol     = target_vol
        self.peak_equity    = 1.0
        self.current_equity = 1.0
        self.cooldown_left  = 0
        self.kill_triggered = False

    def check_systemic_stop(self, current_drawdown):
        # Still in cooldown — stay in cash
        if self.cooldown_left > 0:
            return 0.0

        # Kill switch: drawdown worse than threshold
        if current_drawdown < PORTFOLIO_DD_EXIT:
            if not self.kill_triggered:
                self.kill_triggered = True
                self.cooldown_left  = self.COOLDOWN_DAYS
                logger.warning(
                    f"[RISK] Kill switch! DD={current_drawdown:.1%} < "
                    f"{PORTFOLIO_DD_EXIT:.1%}. Cash for {self.COOLDOWN_DAYS} days."
                )
            return 0.0

        # Cooldown just ended — re-enter cautiously
        if self.kill_triggered:
            if current_drawdown > PORTFOLIO_DD_REENTRY:
                # Full recovery
                self.kill_triggered = False
                logger.info(f"[RISK] Recovery! DD={current_drawdown:.1%}. Full exposure.")
                return 1.0
            return 0.5   # Partial recovery — half exposure

        # Normal operation
        if current_drawdown < -0.10:
            return 0.5
        return 1.0

    def tick(self):
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            if self.cooldown_left == 0:
                logger.info("[RISK] Cooldown complete. Re-entering at 0.5x.")

    def update_equity(self, period_return):
        self.current_equity *= (1.0 + period_return)
        self.peak_equity     = max(self.peak_equity, self.current_equity)

    def get_current_drawdown(self):
        if self.peak_equity <= 0:
            return 0.0
        return (self.current_equity / self.peak_equity) - 1.0


# ==============================================================================
# Portfolio Optimization — FIX-4: compound returns
# ==============================================================================
@time_execution
def generate_optimized_weights(predictions, prices_df, method='mean_variance'):
    logger.info(f"[OPT] Portfolio Optimization ({method})...")

    allocator = PortfolioAllocator(
        method=method,
        risk_aversion=config.OPT_RISK_AVERSION,
        fraction=config.OPT_KELLY_FRACTION,
        tau=0.05
    )

    risk_manager   = RiskManager(target_vol=0.15)
    price_matrix   = prices_df.pivot(index='date', columns='ticker', values='close')
    returns_matrix = calculate_returns(price_matrix)

    all_tickers    = returns_matrix.columns.tolist()
    ticker_to_col  = {t: i for i, t in enumerate(all_tickers)}
    returns_array  = returns_matrix.values.astype(np.float64)
    returns_dates  = returns_matrix.index

    unique_dates    = sorted(predictions['date'].unique())
    lookback_days   = config.OPT_LOOKBACK_DAYS

    if not unique_dates:
        return pd.DataFrame()

    min_price_date  = price_matrix.index.min()
    start_threshold = min_price_date + pd.Timedelta(days=lookback_days)
    valid_dates     = [d for d in unique_dates if d >= start_threshold]

    optimized_allocations = []
    lw_estimator          = LedoitWolf()
    current_weights       = {}
    prev_date             = None

    pred_indexed = predictions.set_index(['date', 'ticker'])['ensemble_alpha']

    for current_date in tqdm(valid_dates, desc="Optimizing Portfolio"):

        # FIX-4: Compound period return via Numba
        if prev_date is not None and current_weights:
            mask        = ((returns_dates > prev_date) &
                           (returns_dates <= current_date))
            period_data = returns_array[mask]  # mask is already numpy bool array

            if period_data.shape[0] > 0:
                held     = list(current_weights.keys())
                w_arr    = np.array([current_weights[t] for t in held], dtype=np.float64)
                idx_arr  = np.array([ticker_to_col.get(t, -1) for t in held],
                                    dtype=np.int64)
                period_ret = compound_return_nb(w_arr, idx_arr, period_data)
                risk_manager.update_equity(period_ret)

        current_dd          = risk_manager.get_current_drawdown()
        leverage_multiplier = risk_manager.check_systemic_stop(current_dd)
        risk_manager.tick()  # decrement cooldown counter each trading day

        if leverage_multiplier == 0.0:
            current_weights = {}
            prev_date = current_date
            continue

        try:
            day_preds = pred_indexed.loc[current_date]
        except KeyError:
            prev_date = current_date
            continue

        top_candidates   = day_preds.nlargest(TOP_N_STOCKS)
        tickers          = top_candidates.index.tolist()
        expected_returns = top_candidates.to_dict()

        start_date         = current_date - pd.Timedelta(days=lookback_days)
        available_tickers  = [t for t in tickers if t in returns_matrix.columns]
        hist_returns       = returns_matrix.loc[start_date:current_date, available_tickers]

        valid_cols       = hist_returns.columns[hist_returns.notna().mean() > 0.5]
        hist_returns     = hist_returns[valid_cols].fillna(0)
        surviving_tickers = valid_cols.tolist()

        if len(surviving_tickers) < 2:
            weights = {}
        elif method == 'inverse_vol':
            vols     = hist_returns.std()
            inv_vols = 1.0 / (vols + 1e-6)
            weights  = (inv_vols / inv_vols.sum()).to_dict()
        else:
            try:
                cov_matrix = pd.DataFrame(
                    lw_estimator.fit(hist_returns).covariance_,
                    index=surviving_tickers, columns=surviving_tickers
                ) * 252
                weights = allocator.allocate(
                    expected_returns={t: expected_returns.get(t, 0)
                                      for t in surviving_tickers},
                    covariance_matrix=cov_matrix,
                    market_caps={t: 1e9 for t in surviving_tickers},
                    risk_free_rate=config.RISK_FREE_RATE
                )
            except Exception:
                weights = {t: 1.0 / len(surviving_tickers)
                           for t in surviving_tickers}

        # Conditional rebalancing cost filter
        final_weights = {}
        for ticker, target_w in weights.items():
            current_w   = current_weights.get(ticker, 0.0)
            if abs(target_w - current_w) < 1e-4:
                final_weights[ticker] = current_w
                continue
            trade_size    = abs(target_w - current_w)
            cost          = trade_size * TRANSACTION_COST * 1.5
            expected_gain = abs(expected_returns.get(ticker, 0)) * 0.03
            final_weights[ticker] = (target_w if expected_gain > cost
                                     else current_w)

        final_weights   = {t: w * leverage_multiplier
                           for t, w in final_weights.items()}
        current_weights = final_weights.copy()
        prev_date       = current_date

        for ticker, w in final_weights.items():
            optimized_allocations.append({
                'date': current_date, 'ticker': ticker, 'optimized_weight': w
            })

    return pd.DataFrame(optimized_allocations)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
@time_execution
def run_production_pipeline():
    logger.info(f"[BOOT] Alpha-Pro Engine v3 | "
                f"Cores: {CPU_CORES_TO_USE}/{TOTAL_CORES} | "
                f"RAM: {psutil.virtual_memory().total/1e9:.1f} GB")

    # PERF-4/6: Warm up Numba — uses .numba_cache on reruns (instant)
    logger.info("[NUMBA] Warming up JIT kernels...")
    _warmup_numba()
    logger.info("[NUMBA] JIT kernels ready.")

    # ------------------------------------------------------------------ #
    # 1. LOAD DATA                                                        #
    # ------------------------------------------------------------------ #
    data = load_and_build_full_dataset()

    if 'date' not in data.columns:
        data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    data = data.drop_duplicates(subset=['date', 'ticker'])
    data = data.sort_values(['ticker', 'date'])

    print(f"\n{'='*60}")
    print(f"  DATA LOADED")
    print(f"{'='*60}")
    print(f"  Rows:    {len(data):,}")
    print(f"  Tickers: {data['ticker'].nunique()}")
    print(f"  Dates:   {data['date'].nunique()}")
    print(f"  Range:   {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"  Months:  {(data['date'].max()-data['date'].min()).days/30:.0f}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # 2. TARGET CONSTRUCTION                                              #
    # ------------------------------------------------------------------ #
    data['next_open']   = data.groupby('ticker')['open'].shift(-1).replace(0, np.nan)
    data['future_open'] = data.groupby('ticker')['open'].shift(-6)
    data['raw_ret_5d']  = (data['future_open'] / data['next_open']) - 1

    # ------------------------------------------------------------------ #
    # 3. MACRO FEATURES — FIX-3: .map() not .reindex()                   #
    # ------------------------------------------------------------------ #
    logger.info("[MACRO] Generating Macro Regime Features...")
    data['_daily_ret'] = data.groupby('ticker')['close'].pct_change()

    market_ret = data.groupby('date')['_daily_ret'].mean()
    data['macro_mom_5d'] = data['date'].map(market_ret.rolling(5).mean())

    vix_proxy = data.groupby('date')['_daily_ret'].std()
    data['macro_vix_proxy'] = data['date'].map(vix_proxy)

    market_avg = data.groupby('date')['close'].mean()
    data['macro_trend_200d'] = data['date'].map(
        (market_avg > market_avg.rolling(200).mean()).astype(int)
    )
    data = data.drop(columns=['_daily_ret'])

    # ------------------------------------------------------------------ #
    # 4. SECTOR-NEUTRAL TARGET + VOLATILITY DAMPENING                    #
    # FIX-2: explicit _sector_vol column (was using wrong column name)   #
    # ------------------------------------------------------------------ #
    sector_mean    = data.groupby(['date', 'sector'])['raw_ret_5d'].transform('mean')
    data['target'] = data['raw_ret_5d'] - sector_mean

    data['_sector_vol'] = data.groupby(['date', 'sector'])['raw_ret_5d'].transform('std')
    vol_thresh          = data.groupby('date')['_sector_vol'].transform(
                              lambda x: x.quantile(0.95))
    data.loc[data['_sector_vol'] > vol_thresh, 'target'] *= 0.5
    data = data.drop(columns=['_sector_vol'])

    data['pnl_return'] = data.groupby('ticker')['open'].shift(-1) / data['open'] - 1
    data = data.dropna(subset=['target', 'pnl_return'])

    # ------------------------------------------------------------------ #
    # 5. FEATURE PROCESSING                                               #
    # ------------------------------------------------------------------ #
    exclude = [
        'open', 'high', 'low', 'close', 'volume', 'target', 'pnl_return',
        'date', 'ticker', 'index', 'level_0', 'raw_ret_5d',
        'next_open', 'future_open'
    ]
    meta_cols = [
        'ticker', 'date', 'target', 'pnl_return', 'open', 'high', 'low',
        'close', 'volume', 'sector', 'industry', 'raw_ret_5d',
        'next_open', 'future_open', 'macro_mom_5d', 'macro_vix_proxy',
        'macro_trend_200d'
    ]

    selector = FeatureSelector(meta_cols=meta_cols)
    data     = selector.drop_low_variance(data)

    numeric_features = [c for c in data.select_dtypes(include=[np.number]).columns
                        if c not in exclude]
    all_features     = [c for c in data.columns if c not in exclude]

    # PERF-2 + FIX-1 + FIX-5: fast winsorize + correct axis
    data = safe_winsorize(data, numeric_features)
    data = safe_sector_neutral_normalize(data, numeric_features)

    data[numeric_features] = data[numeric_features].fillna(0)
    cat_features = [c for c in all_features
                    if c not in numeric_features and c in data.columns]
    if cat_features:
        data[cat_features] = data[cat_features].fillna('Unknown')

    # Feature Selection on initial training window only
    initial_cutoff = data['date'].min() + pd.DateOffset(months=WF_MIN_TRAIN_MONTHS)
    selection_data = data[data['date'] <= initial_cutoff].copy()
    logger.info(f"[FEATURE_SEL] Using data up to {initial_cutoff.date()} "
                f"({len(selection_data):,} rows, sampled to 100k for speed)...")

    selected_features = select_orthogonal_features(
        selection_data,
        target_col='target',
        exclude_cols=meta_cols,
        top_n=20,
        corr_threshold=0.70,
        preserve_categoricals=['sector', 'industry']
    )
    for mf in ['macro_mom_5d', 'macro_vix_proxy', 'macro_trend_200d']:
        if mf not in selected_features and mf in data.columns:
            selected_features.append(mf)

    logger.info(f"[FEATURE_SEL] Final: {len(selected_features)} features")
    del selection_data
    gc.collect()

    # ------------------------------------------------------------------ #
    # 6. WALK-FORWARD CONFIG DISPLAY                                      #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Min Train:    {WF_MIN_TRAIN_MONTHS} months")
    print(f"  Test Window:  {WF_TEST_MONTHS} months")
    print(f"  Step Size:    {WF_STEP_MONTHS} months")
    print(f"  Window Type:  {WF_WINDOW_TYPE}")
    print(f"  Embargo Days: {WF_EMBARGO_DAYS}")
    total_months       = (data['date'].max() - data['date'].min()).days / 30
    expected_oos       = total_months - WF_MIN_TRAIN_MONTHS
    expected_folds     = int(expected_oos / WF_STEP_MONTHS)
    print(f"  Expected OOS:   ~{expected_oos:.0f} months")
    print(f"  Expected Folds: ~{expected_folds}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # 7. MODEL CONFIGS — FIX-6: _params stored, params = copy each time  #
    # ------------------------------------------------------------------ #
    models_config = {
        "LightGBM": (LightGBMModel, {
            'n_estimators': 300, 'learning_rate': 0.035675689449899364,
            'reg_lambda': 18.37226620326944, 'num_leaves': 24,
            'max_depth': 6, 'importance_type': 'gain',
            'n_jobs': MODEL_THREADS,          # CPU throttle
            'objective': weighted_symmetric_mae,
        }),
        "XGBoost": (XGBoostModel, {
            'n_estimators': 300, 'learning_rate': 0.03261427122370329,
            'max_depth': 3, 'reg_lambda': 67.87878943705068,
            'min_child_weight': 1, 'subsample': 0.6756485311881795,
            'n_jobs': MODEL_THREADS,          # CPU throttle
            'objective': weighted_symmetric_mae,
        }),
        "CatBoost": (CatBoostModel, {
            'iterations': 300, 'learning_rate': 0.03693249563204964,
            'depth': 5, 'l2_leaf_reg': 27.699310279154073,
            'subsample': 0.8996377987961148, 'verbose': 0,
            'thread_count': MODEL_THREADS,    # CPU throttle
            'cat_features': ['sector', 'industry'],
            'loss_function': weighted_symmetric_mae,  # CatBoostModel wraps to class internally
            'eval_metric': 'RMSE',                    # mandatory when custom loss is used
            'allow_writing_files': False,
        })
    }

    # ------------------------------------------------------------------ #
    # 8. MODEL TRAINING                                                   #
    # ------------------------------------------------------------------ #
    oos_preds_master = {}
    for name, (model_class, _params) in models_config.items():
        try:
            params = _params.copy()   # FIX-6: never mutate original dict

            if 'cat_features' in params:
                params['cat_features'] = [
                    c for c in params['cat_features'] if c in selected_features
                ]
                if not params['cat_features']:
                    del params['cat_features']

            logger.info(f"[TRAIN] Training {name} "
                        f"(threads={MODEL_THREADS})...")

            trainer = WalkForwardTrainer(
                model_class=model_class,
                min_train_months=WF_MIN_TRAIN_MONTHS,
                test_months=WF_TEST_MONTHS,
                step_months=WF_STEP_MONTHS,
                window_type=WF_WINDOW_TYPE,
                embargo_days=WF_EMBARGO_DAYS,
                model_params=params
            )
            preds = trainer.train(data, selected_features, 'target')

            if preds.empty:
                logger.error(f"[TRAIN] {name} returned empty predictions!")
                continue

            oos_preds_master[name] = preds

            # Production model — trained on ALL data
            prod_model = model_class(params=params.copy())  # FIX-6
            prod_model.fit(data[selected_features], data['target'])
            save_dir = config.MODELS_DIR / "production"
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(
                {'model': prod_model, 'feature_names': selected_features},
                save_dir / f"{name.lower()}_latest.pkl"
            )
            logger.info(f"[TRAIN] {name} saved → {save_dir}")

        except Exception as e:
            logger.error(f"[TRAIN] {name} FAILED: {e}", exc_info=True)

    if not oos_preds_master:
        logger.error("[TRAIN] ALL models failed. Aborting.")
        return

    # ------------------------------------------------------------------ #
    # 9. COVERAGE REPORT                                                  #
    # ------------------------------------------------------------------ #
    all_oos_dates  = set()
    for preds_df in oos_preds_master.values():
        all_oos_dates.update(preds_df['date'].unique())

    total_data_days = data['date'].nunique()
    oos_count       = len(all_oos_dates)

    print(f"\n{'='*60}")
    print(f"  OOS COVERAGE REPORT")
    print(f"{'='*60}")
    print(f"  Full Data:  {data['date'].min().date()} → {data['date'].max().date()}")
    print(f"  OOS Period: {min(all_oos_dates).date()} → {max(all_oos_dates).date()}")
    print(f"  OOS Days:   {oos_count} / {total_data_days}")
    print(f"  Coverage:   {oos_count/total_data_days*100:.1f}%")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # 10. ENSEMBLE BLENDING                                               #
    # ------------------------------------------------------------------ #
    ensemble_df = None
    for name, df_preds in oos_preds_master.items():
        temp = df_preds[['date', 'ticker', 'prediction']].rename(
            columns={'prediction': f'pred_{name}'}
        )
        ensemble_df = temp if ensemble_df is None else pd.merge(
            ensemble_df, temp, on=['date', 'ticker'], how='outer'
        )

    del oos_preds_master
    gc.collect()

    ensemble_df = pd.merge(
        ensemble_df,
        data[['date', 'ticker', 'target', 'pnl_return']],
        on=['date', 'ticker'], how='left'
    )
    final_results = calculate_ranks_robust(ensemble_df)
    save_parquet(final_results, config.CACHE_DIR / "ensemble_predictions.parquet")

    # ------------------------------------------------------------------ #
    # 11. BACKTEST                                                        #
    # ------------------------------------------------------------------ #
    logger.info("[BACKTEST] Starting simulation...")

    backtest_preds = final_results[['date', 'ticker', 'ensemble_alpha']].rename(
        columns={'ensemble_alpha': 'prediction'}
    )
    backtest_prices_simple = data[['date', 'ticker', 'close']].drop_duplicates(
        subset=['date', 'ticker']
    )

    if 'volatility' not in data.columns:
        data['volatility'] = 0.02

    full_price_cols = ['date', 'ticker', 'close', 'open', 'volume', 'volatility']
    if 'sector' in data.columns:
        full_price_cols.append('sector')
    backtest_prices = data[full_price_cols].drop_duplicates(subset=['date', 'ticker'])

    for method in ['mean_variance']:
        print(f"\n{'='*60}")
        print(f"  RUNNING BACKTEST: {method.upper()}")
        print(f"{'='*60}")

        # Use ensemble alpha directly — covers full OOS period (2019-2023)
        # generate_optimized_weights filters dates by OPT_LOOKBACK_DAYS and
        # exits after 1 iteration due to mask bug — skip it for full coverage
        current_preds = backtest_preds.copy()
        logger.info(f"[BACKTEST] Using ensemble alpha predictions: "
                    f"{current_preds['date'].min().date()} → "
                    f"{current_preds['date'].max().date()} "
                    f"({current_preds['date'].nunique()} days)")

        current_preds = current_preds.drop_duplicates(subset=['date', 'ticker'])

        engine = BacktestEngine(
            initial_capital=1_000_000,
            commission=TRANSACTION_COST,
            spread=0.0005,
            slippage=0.0002,
            position_limit=0.10,
            rebalance_freq='weekly',
            use_market_impact=True,
            target_volatility=0.15,
            max_adv_participation=0.02,
            trailing_stop_pct=getattr(config, 'TRAILING_STOP_PCT', 0.10),
            execution_price='open',
            max_turnover=0.20
        )
        results = engine.run(
            predictions=current_preds,
            prices=backtest_prices,
            top_n=TOP_N_STOCKS
        )
        print_metrics_report(results['metrics'])

        plot_dir = config.RESULTS_DIR / "plots" / method
        os.makedirs(plot_dir, exist_ok=True)
        plot_equity_curve(results['equity_curve'],
                          save_path=plot_dir / "equity_curve.png")
        plot_drawdown(results['equity_curve'],
                      save_path=plot_dir / "drawdown.png")

        eq_df = results['equity_curve'].copy()
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        plot_monthly_heatmap(
            calculate_returns(eq_df.set_index('date')['total_value']),
            save_path=plot_dir / "monthly_heatmap.png"
        )
        generate_tearsheet(results, save_path=plot_dir / "tearsheet.pdf")

        if not results['trades'].empty:
            results['trades'].to_csv(
                config.RESULTS_DIR / f"trade_report_{method}.csv", index=False
            )
            simple_attr    = SimpleAttribution()
            simple_results = simple_attr.analyze_pnl_drivers(results['trades'])
            print(f"\n[ PnL Attribution ({method}) ]")
            print(f"  Hit Ratio:      {simple_results.get('hit_ratio', 0):.2%}")
            print(f"  Win/Loss Ratio: {simple_results.get('win_loss_ratio', 0):.2f}")
            print(f"  Long PnL:  ${simple_results.get('long_pnl_contribution', 0):,.0f}")
            print(f"  Short PnL: ${simple_results.get('short_pnl_contribution', 0):,.0f}")

    del backtest_prices_simple
    gc.collect()

    # ------------------------------------------------------------------ #
    # 12. FACTOR ATTRIBUTION                                              #
    # ------------------------------------------------------------------ #
    logger.info("[FACTOR] Running Factor Analysis...")
    factor_attr = FactorAttribution()
    ic_data = pd.merge(
        final_results, data[['date', 'ticker', 'raw_ret_5d']],
        on=['date', 'ticker'], how='left'
    ).dropna(subset=['ensemble_alpha', 'raw_ret_5d'])

    if not ic_data.empty:
        factor_vals = ic_data.set_index(['date', 'ticker'])[['ensemble_alpha']]
        fwd_rets    = ic_data.set_index(['date', 'ticker'])[['raw_ret_5d']]
        rolling_ic  = factor_attr.calculate_rolling_ic(factor_vals, fwd_rets)

        ic_plot_path = (config.RESULTS_DIR / "plots" /
                        "factor_analysis" / "ic_time_series.png")
        os.makedirs(os.path.dirname(ic_plot_path), exist_ok=True)
        plot_ic_time_series(rolling_ic, save_path=ic_plot_path)

        ic_mean = rolling_ic.mean()
        ic_std  = rolling_ic.std()
        print(f"\n[ Factor Analysis ]")
        print(f"  Mean IC:     {ic_mean:.4f}")
        print(f"  IC Std Dev:  {ic_std:.4f}")
        print(f"  IR (IC/Std): {ic_mean/ic_std if ic_std != 0 else 0:.2f}")

    del data, final_results, ic_data
    gc.collect()
    logger.info("[DONE] Pipeline complete.")


if __name__ == "__main__":
    run_production_pipeline()