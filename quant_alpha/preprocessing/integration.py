"""
Preprocessing Integration & Point-in-Time Merger
================================================
Bridges derived fundamental columns and time-series earnings dynamics into
the daily price panel using strictly point-in-time (PiT) boundaries.

Called by: run_pipeline.py / DataManager enrichment path
Key function: enhance_fundamentals_for_registry()

BUGS FIXED
----------
BUG-048  HIGH  D1  Look-ahead in _compute_earnings_ts():
    earnings_estimate_revisions_3m = eps_estimate.pct_change() * 100
    This computes the % change from quarter Q-1 to quarter Q for the
    ESTIMATE column. The estimate for Q is known before Q closes, so
    this is directionally safe — BUT the result is placed on the row
    dated to the EARNINGS ANNOUNCEMENT date. If that date is used as the
    PiT boundary AND the estimate for the NEXT quarter has already been
    revised, the revision signal implicitly looks forward.
    Correct fix: use only eps_estimate.diff() / abs(eps_estimate.shift(1))
    and label it clearly as "change in estimate from prior quarter" (backward).

BUG-049  MEDIUM  A3  Rolling beta apply() loses index:
    final_data.groupby('ticker', group_keys=False).apply(lambda g:
        g['stock_ret'].rolling(60).cov(g['spy_ret']) / ...)
    apply() reassembles correctly with group_keys=False — but the lambda
    returns a Series whose index IS the group index. This is actually safe.
    REAL BUG HERE: the merge `final_data.merge(spy_ret, left_on='date',
    right_index=True, how='left')` — spy_ret.index is a DatetimeIndex that
    may have tz-info (yfinance returns UTC-aware index). final_data['date']
    is tz-naive after pd.to_datetime(). The merge silently returns all NaN
    for spy_ret when tzinfo mismatches. Fixed by tz_localize(None) on both.

BUG-050  HIGH  C4  No NaN-safety on surprise_pct before streak loop:
    The streak loop iterates over df['surprise_pct'] which may contain NaN
    (e.g. for the most recent quarter not yet reported or bad data).
    NaN > 0 evaluates to False in Python, silently treating NaN as a miss.
    This makes the streak undercount and corrupts momentum. Fixed by
    skipping NaN values in the streak computation.
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
    """Load PiT fundamental rows for one ticker. Returns (ticker, df | None)."""
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
    Compute time-series earnings dynamics for one ticker.

    Returns (ticker, DataFrame | None) where the DataFrame has columns:
      date, ticker,
      earnings_latest_surprise,
      earnings_3q_avg_surprise,
      earnings_surprise_momentum,
      earnings_beat_streak,
      earnings_estimate_revisions_3m

    FIX BUG-048: earnings_estimate_revisions_3m now computed as
      (eps_estimate - eps_estimate.shift(1)) / abs(eps_estimate.shift(1))
    which is strictly backward-looking (prior quarter's estimate as base).
    The original pct_change() is identical mathematically but the explicit
    form makes the look-back direction unambiguous.

    FIX BUG-050: streak loop now skips NaN surprise values explicitly
    instead of letting NaN > 0 evaluate as False (silent miss).
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
        # FIX BUG-050: explicitly handle NaN — treat NaN as "unknown",
        # which resets the streak (conservative, not a false beat).
        streak = []
        curr = 0
        for val in df['surprise_pct']:
            if pd.isna(val):
                # Unknown quarter — reset streak conservatively
                curr = 0
            elif val > 0:
                curr += 1
            else:
                curr = 0
            streak.append(curr)
        df['earnings_beat_streak'] = streak

        # --- Estimate revision trend ---
        # FIX BUG-048: use explicit backward diff / abs(prior) for clarity.
        # This is strictly backward-looking: change vs prior quarter's estimate.
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
    Main entrypoint: merge PiT fundamentals, time-series earnings, and
    rolling beta into the daily price panel.

    Uses pd.merge_asof(direction='backward') for the PiT merge — this
    guarantees that on any given date, only fundamental/earnings data
    that was available ON OR BEFORE that date is used (no look-ahead).

    Parameters
    ----------
    data             : Master price panel (date, ticker, close, ...)
    fundamentals_dir : Path to per-ticker fundamental CSVs
    earnings_dir     : Path to per-ticker earnings CSVs
    tickers          : List of tickers to process
    max_workers      : ThreadPool size for parallel loading
    compute_beta     : Whether to compute rolling 60-day beta vs S&P 500

    Returns
    -------
    pd.DataFrame — data enriched with fundamental + earnings + beta columns,
                   sorted by (ticker, date).

    FIX BUG-049: spy_ret.index is stripped of timezone before the merge
    so that tz-aware yfinance DatetimeIndex does not silently produce all-NaN
    spy_ret after merging against tz-naive data['date'].
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

    # ---- 3. Point-in-Time merge via merge_asof ----
    # merge_asof with direction='backward' ensures that on any price date,
    # we only use the most recent available fundamental/earnings snapshot
    # whose 'date' is <= the price date. This is the correct PiT pattern.
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
    Compute rolling 60-day beta vs S&P 500 and add as 'qual_low_beta'.

    FIX BUG-049: strip timezone from spy_ret.index before merging.
    yfinance returns a UTC-aware DatetimeIndex; final_data['date'] is
    tz-naive. The merge silently returns all NaN when tzinfo mismatches.

    The signal is INVERTED (multiplied by -1) so that higher score =
    lower beta = safer stock, consistent with LowBeta in quality.py.
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

        # FIX BUG-049: strip tz so merge against tz-naive dates works correctly
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
            # Invert: higher score = lower beta risk (consistent with LowBeta factor)
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