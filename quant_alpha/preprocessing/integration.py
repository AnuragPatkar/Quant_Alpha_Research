"""
Preprocessing Integration & Point-in-Time Merger
================================================

Bridges derived fundamental columns and time-series earnings dynamics into
the daily continuous price panel utilizing strictly Point-in-Time (PiT) boundaries.

Purpose
-------
This module strictly aligns low-frequency structural data (quarterly/annual 
fundamentals) with high-frequency market data (daily OHLCV) without inducing 
look-ahead bias.

Role in Quantitative Workflow
-----------------------------
Serves as the primary data fusion engine for the research pipeline. Employs 
backward-looking asynchronous merges to guarantee that predictive algorithms 
only evaluate information that was historically public on any given execution date.

Mathematical Dependencies
-------------------------
- **Pandas**: Employs `merge_asof` operations for $O(N \log N)$ temporal alignments 
  and expanding rolling covariance bounds for asset beta estimations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm

from .fundamental_preprocessor import preprocess_fundamentals

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_single_ticker_fundamentals(args):
    """
    Executes parallel extraction mapping strictly localized PiT fundamental rows per individual ticker.

    Args:
        args (tuple): A structural tuple bounding the precise `ticker` (str) and `fund_dir` (Path).

    Returns:
        Tuple[str, Optional[pd.DataFrame]]: The explicit corresponding identifier and discrete asset dataframe.
    """
    ticker, fund_dir = args
    try:
        fund_df = preprocess_fundamentals(fund_dir, ticker)
        if fund_df is not None and not fund_df.empty:
            fund_df = fund_df.sort_values('date')
        return ticker, fund_df
    except Exception as e:
        logger.debug(f"[Integration] Fundamentals failed for {ticker}: {e}")
        return ticker, None


def _compute_earnings_ts(ticker: str, earnings_dir) -> tuple:
    """
    Computes standardized time-series earnings dynamics mapped continuously for a single asset.

    Args:
        ticker (str): The discrete targeting equity identifier string.
        earnings_dir (Path): The foundational source directory mapping localized earnings events.

    Returns:
        Tuple[str, Optional[pd.DataFrame]]: Standardized tuple matching the target ticker and its structured 
            earnings trajectory derivations.
    """
    earn_file = Path(earnings_dir) / f"{ticker}.csv"
    if not earn_file.exists():
        return ticker, None

    try:
        df = pd.read_csv(earn_file)
        if 'date' not in df.columns or df.empty:
            return ticker, None

        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        df = df.sort_values('date').dropna(subset=['eps_actual']).reset_index(drop=True)

        if df.empty:
            return ticker, None

        # --- Latest surprise ---
        df['earnings_latest_surprise'] = df['surprise_pct']

        # --- 3-quarter rolling average ---
        df['earnings_3q_avg_surprise'] = (
            df['surprise_pct'].rolling(3, min_periods=1).mean()
        )

        # --- Surprise momentum: recent 3Q avg vs prior 3Q avg ---
        recent = df['surprise_pct'].rolling(3, min_periods=1).mean()
        prior  = df['surprise_pct'].shift(3).rolling(3, min_periods=1).mean()
        df['earnings_surprise_momentum'] = recent - prior

        # --- Beat streak ---
        # Explicitly handles NaN boundaries by treating them as mathematically "unknown",
        # which strictly resets the rolling consecutive streak to enforce a conservative state.
        streak = []
        curr = 0
        for val in df['surprise_pct']:
            if pd.isna(val):
                curr = 0
            elif val > 0:
                curr += 1
            else:
                curr = 0
            streak.append(curr)
        df['earnings_beat_streak'] = streak

        # --- Estimate revision trend ---
        # Computes continuous structural drift strictly against the prior trailing quarter's 
        # estimate base to categorically prevent look-ahead target leakage.
        prior_est = df['eps_estimate'].shift(1)
        df['earnings_estimate_revisions_3m'] = (
            (df['eps_estimate'] - prior_est) / (prior_est.abs() + 1e-9) * 100
        )

        df['ticker'] = ticker

        out_cols = [
            'date', 'ticker',
            'earnings_latest_surprise',
            'earnings_3q_avg_surprise',
            'earnings_surprise_momentum',
            'earnings_beat_streak',
            'earnings_estimate_revisions_3m',
        ]
        return ticker, df[out_cols]

    except Exception as e:
        logger.debug(f"[Integration] Earnings TS failed for {ticker}: {e}")
        return ticker, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enhance_fundamentals_for_registry(
    data: pd.DataFrame,
    fundamentals_dir: Path,
    earnings_dir: Path,
    tickers: list,
    max_workers: int = 8,
    compute_beta: bool = True,
) -> pd.DataFrame:
    """
    Primary integration gateway merging Point-in-Time fundamentals, continuous earnings 
    trajectories, and trailing rolling beta into the localized price panel.

    Args:
        data (pd.DataFrame): Primary daily pricing matrix containing core OHLCV bounds.
        fundamentals_dir (Path): Source directory mapping static preprocessed fundamental CSVs.
        earnings_dir (Path): Source directory mapping strictly aligned earnings histories.
        tickers (list): Explicit ticker list defining the processing universe.
        max_workers (int): Constrains concurrent thread pool extraction parallelization. Defaults to 8.
        compute_beta (bool): Determines if dynamic 60-day relative variance against the 
            S&P 500 should be appended to the panel. Defaults to True.

    Returns:
        pd.DataFrame: A unified temporal DataFrame strictly sorted by (ticker, date) containing 
            all enriched PiT characteristics.
    """
    logger.info(
        f"[Integration] Enhancing {len(tickers)} tickers with "
        "PiT fundamentals and earnings..."
    )

    # ---- 1. Load PiT fundamentals in parallel ----
    all_fund = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exc:
        futures = {
            exc.submit(_process_single_ticker_fundamentals, (t, fundamentals_dir)): t
            for t in tickers
        }
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(tickers),
            desc="PiT Fundamentals",
        ):
            _, f_df = fut.result()
            if f_df is not None and not f_df.empty:
                all_fund.append(f_df)

    fund_panel = (
        pd.concat(all_fund, ignore_index=True)
        if all_fund
        else pd.DataFrame()
    )
    if not fund_panel.empty:
        fund_panel['date'] = pd.to_datetime(fund_panel['date'], utc=True).dt.tz_localize(None)
        logger.info(
            f"[Integration] Fundamental panel: {len(fund_panel):,} rows, "
            f"{fund_panel['ticker'].nunique()} tickers."
        )

    # ---- 2. Load earnings time-series in parallel ----
    all_earn = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exc:
        futures = {
            exc.submit(_compute_earnings_ts, t, earnings_dir): t
            for t in tickers
        }
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(tickers),
            desc="TS Earnings",
        ):
            _, e_df = fut.result()
            if e_df is not None and not e_df.empty:
                all_earn.append(e_df)

    earn_panel = (
        pd.concat(all_earn, ignore_index=True)
        if all_earn
        else pd.DataFrame()
    )
    if not earn_panel.empty:
        earn_panel['date'] = pd.to_datetime(earn_panel['date'], utc=True).dt.tz_localize(None)
        logger.info(
            f"[Integration] Earnings panel: {len(earn_panel):,} rows, "
            f"{earn_panel['ticker'].nunique()} tickers."
        )

    # ---- 3. Asynchronous Point-in-Time Temporal Merge ----
    # Enforces explicit direction='backward' constraints guaranteeing any executing date
    # strictly incorporates identical available information mapped sequentially <= the price vector.
    # Mathematically resolves potential intersection discrepancies scaling data loops.
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'], utc=True).dt.tz_localize(None)
    data = data.sort_values(['ticker', 'date']).reset_index(drop=True)

    merged_chunks = []
    for ticker, grp in tqdm(
        data.groupby('ticker', sort=False),
        desc="PiT Merge",
        total=data['ticker'].nunique(),
    ):
        chunk = grp.sort_values('date')

        if not fund_panel.empty:
            t_fund = (
                fund_panel[fund_panel['ticker'] == ticker]
                .drop(columns=['ticker'])
                .sort_values('date')
            )
            if not t_fund.empty:
                chunk = pd.merge_asof(
                    chunk, t_fund,
                    on='date',
                    direction='backward',
                )

        if not earn_panel.empty:
            t_earn = (
                earn_panel[earn_panel['ticker'] == ticker]
                .drop(columns=['ticker'])
                .sort_values('date')
            )
            if not t_earn.empty:
                chunk = pd.merge_asof(
                    chunk, t_earn,
                    on='date',
                    direction='backward',
                )

        merged_chunks.append(chunk)

    if not merged_chunks:
        logger.warning("[Integration] No chunks merged — returning original data.")
        return data

    final_data = pd.concat(merged_chunks, ignore_index=True)
    logger.info(
        f"[Integration] PiT merge complete: "
        f"{final_data.shape[0]:,} rows × {final_data.shape[1]} cols."
    )

    # ---- 4. Rolling 60-day Beta vs S&P 500 ----
    if compute_beta:
        final_data = _add_rolling_beta(final_data)

    final_data = (
        final_data
        .sort_values(['ticker', 'date'])
        .reset_index(drop=True)
    )
    return final_data


def _add_rolling_beta(final_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes expanding 60-day rolling covariance beta against the benchmark S&P 500.

    Constructs the relative structural beta coefficient and systematically inverts the 
    scalar derivative ($score = -1 \times \beta$) ensuring that ascending boundary scores 
    correlate uniformly with lower historical systemic risk profiles.

    Args:
        final_data (pd.DataFrame): The unified asset matrix targeting beta augmentation.

    Returns:
        pd.DataFrame: The equivalent matrix embedded with the derived 'qual_low_beta' vector.
    """
    try:
        import yfinance as yf

        start_dt = final_data['date'].min().strftime('%Y-%m-%d')
        end_dt   = final_data['date'].max().strftime('%Y-%m-%d')

        logger.info(
            f"[Integration] Computing 60-day rolling Beta "
            f"({start_dt} → {end_dt})..."
        )

        spy = yf.download(
            '^GSPC',
            start=start_dt,
            end=end_dt,
            progress=False,
            auto_adjust=True,
        )

        if spy.empty:
            logger.warning("[Integration] yfinance returned empty SPY data.")
            final_data['qual_low_beta'] = np.nan
            return final_data

        spy_col = spy['Close'] if 'Close' in spy.columns else spy.iloc[:, 0]
        spy_ret = spy_col.pct_change().rename('spy_ret')

        # Strips UTC timezone localization parameters from external indexing vectors 
        # to ensure strictly compliant temporal merges against naive execution dates.
        spy_ret.index = pd.to_datetime(spy_ret.index).tz_localize(None)

        final_data = final_data.merge(
            spy_ret,
            left_on='date',
            right_index=True,
            how='left',
        )
        final_data['stock_ret'] = (
            final_data.groupby('ticker')['close'].pct_change()
        )

        def _rolling_beta(g: pd.DataFrame) -> pd.Series:
            cov = g['stock_ret'].rolling(60, min_periods=30).cov(g['spy_ret'])
            var = g['spy_ret'].rolling(60, min_periods=30).var()
            beta = cov / (var + 1e-9)
            
            return -1.0 * beta

        final_data['qual_low_beta'] = (
            final_data
            .groupby('ticker', group_keys=False)
            .apply(_rolling_beta)
        )
        final_data = final_data.drop(columns=['spy_ret', 'stock_ret'])

        n_valid = final_data['qual_low_beta'].notna().sum()
        logger.info(
            f"[Integration] Beta computed: {n_valid:,} valid rows."
        )

    except ImportError:
        logger.warning("[Integration] yfinance not installed — skipping beta.")
        final_data['qual_low_beta'] = np.nan
    except Exception as e:
        logger.warning(f"[Integration] Failed to compute beta: {e}")
        final_data['qual_low_beta'] = np.nan

    return final_data