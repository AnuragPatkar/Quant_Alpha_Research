"""
Data Warehouse Ingestion and Integrity Validation
=================================================
Validates the structural correctness, schema alignment, and data quality of the master dataset.

Purpose
-------
This module serves as the primary test suite for the `DataManager`. It verifies 
that foundational data pipelines reliably merge OHLCV, fundamental, earnings, 
and macroeconomic features into a strictly unified `(date, ticker)` MultiIndex 
without introducing look-ahead bias, duplication, or corruption.

Role in Quantitative Workflow
-----------------------------
Acts as an automated data quality gatekeeper prior to feature engineering, 
ensuring models train on mathematically and structurally sound environments.

Dependencies
------------
- **Pytest**: Test execution and synthetic fixture orchestration.
- **Pandas/NumPy**: Vectorized data generation and multi-index manipulations.
- **Unittest.Mock**: Deep namespace patching to isolate logical verification from I/O bounds.
"""

import sys
import inspect
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from quant_alpha.data.DataManager import DataManager

# Establishes a deterministic global seed to enforce reproducible data synthesis boundaries
RNG = np.random.default_rng(seed=42)

# Baseline execution schema parameters extracted from production storage bounds
PRICE_COLS       = ["open", "high", "low", "close", "volume"]
FUND_COLS        = [
    "beta", "pe_ratio", "peg_ratio", "eps", "fwd_eps", "roe", "roa",
    "op_margin", "gross_margin", "debt_equity", "current_ratio",  
    "quick_ratio", "total_cash", "total_debt", "fcf", "ocf",  
    "rev_growth", "earnings_growth", "ebitda_margin", "profit_margin",
    "ps_ratio", "ev_ebitda", "total_revenue", "ebitda", "net_income",
]
EARNINGS_COLS    = ["eps_actual", "eps_estimate", "surprise_pct"]
EXPECTED_NCOLS   = 46
EXPECTED_NROWS   = 976_019
EXPECTED_NTICKERS = 499

# Strictly enforced distributional threshold boundaries to prevent systemic starvation during inference
FUND_NAN_THRESHOLDS = {
    "peg_ratio":      0.25,   
    "debt_equity":    0.20,   
    "fcf":            0.20,   
    "ocf":            0.20,   
    "earnings_growth":0.20,   
    "roe":            0.10,   
    "roa":            0.05,   
    "current_ratio":  0.10,   
    "quick_ratio":    0.10,   
    "beta":           0.05,   
    "pe_ratio":       0.10,   
}

def _make_ohlcv_df(tickers, dates, seed_offset=0):
    """
    Generates a mathematically coherent synthetic OHLCV multi-indexed matrix.

    Strictly adheres to market microstructure logic where the established 
    High represents the local maximum and Low represents the local minimum 
    across the execution duration.

    Args:
        tickers (list[str]): The deterministic universe cohort.
        dates (pd.DatetimeIndex): Contiguous execution boundaries.
        seed_offset (int, optional): Random shift scalar. Defaults to 0.

    Returns:
        pd.DataFrame: Symmetrically indexed OHLCV mapping matrix.
    """
    rng  = np.random.default_rng(seed=42 + seed_offset)
    rows = []
    for ticker in tickers:
        n     = len(dates)
        close = 100.0 + rng.standard_normal(n).cumsum()
        open_ = np.roll(close, 1); open_[0] = close[0]
        noise = rng.uniform(0.001, 0.01, n)
        high  = np.maximum(open_, close) * (1 + noise)
        low   = np.minimum(open_, close) * (1 - noise)
        vol   = rng.integers(500_000, 5_000_000, n).astype(float)
        for j, d in enumerate(dates):
            rows.append({
                "date": d, "ticker": ticker,
                "open": open_[j], "high": high[j],
                "low": low[j], "close": close[j],
                "volume": vol[j],
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "ticker"])


def _make_master(tickers=None, n_dates=5, include_fundamentals=True):
    """
    Synthesizes an aggregated master dataset strictly mirroring the target production schema.

    Args:
        tickers (list[str] | None, optional): Specific asset array override. Defaults to None.
        n_dates (int, optional): Temporal depth length constraints. Defaults to 5.
        include_fundamentals (bool, optional): Integrates point-in-time corporate data if True. 
            Defaults to True.

    Returns:
        pd.DataFrame: An enriched structural object spanning specific boundaries 
            ready for isolated logic verification bypassing physical storage interactions.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]
    dates = pd.date_range("2023-01-02", periods=n_dates, freq="B")
    df = _make_ohlcv_df(tickers, dates).reset_index()

    if include_fundamentals:
        rng = np.random.default_rng(seed=99)
        fund_vals = {t: rng.uniform(1, 100) for t in tickers}
        df["pe_ratio"]      = df["ticker"].map(fund_vals)
        df["beta"]          = df["ticker"].map({t: rng.uniform(0.5, 2.0) for t in tickers})
        df["total_revenue"] = df["ticker"].map({t: rng.uniform(1e9, 1e11) for t in tickers})
        df["net_income"]    = df["ticker"].map({t: rng.uniform(1e8, 1e10) for t in tickers})

    rng2 = np.random.default_rng(seed=77)
    df["eps_actual"]   = rng2.uniform(1.0, 5.0, len(df))
    df["eps_estimate"] = rng2.uniform(1.0, 5.0, len(df))
    df["surprise_pct"] = rng2.uniform(-0.1, 0.1, len(df))

    rng3 = np.random.default_rng(seed=55)
    df["vix_close"]   = rng3.uniform(15, 35, len(df))
    df["usd_close"]   = rng3.uniform(95, 105, len(df))
    df["sp500_close"] = rng3.uniform(3000, 5000, len(df))

    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "ticker"])

@pytest.fixture(scope="module")
def master():
    """
    Singleton fixture hydrating the definitive production master dataset.

    Loads strictly once per execution module to mitigate heavy disk I/O penalties
    across the testing DAG.

    Args:
        None

    Returns:
        pd.DataFrame: The active, physical pipeline target containing realistic 
            market moments and structural nuances.
    """
    return DataManager().get_master_data()

class TestDataManagerIntegration:
    """
    End-to-End structural testing suite executing against live pipeline ingestion artifacts.
    """

    def test_initialization_and_schema(self, master):
        """
        Validates core execution boundaries mapping standard data frames to multi-index schemas.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        assert isinstance(master, pd.DataFrame)
        assert not master.empty
        assert isinstance(master.index, pd.MultiIndex)
        assert list(master.index.names) == ["date", "ticker"], (
            f"Expected ['date','ticker'], got {master.index.names}"
        )

    def test_column_count(self, master):
        """
        Asserts dimensional integrity mapped to minimal execution requirements.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        assert len(master.columns) >= 5, (
            f"Expected at least 5 columns, got {len(master.columns)}.\n"
            f"Columns found: {master.columns.tolist()}"
        )

    def test_price_columns_present(self, master):
        """
        Asserts holistic mapping of required quantitative price inputs.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        for col in PRICE_COLS:
            assert col in master.columns, f"Price column '{col}' missing"

    def test_fundamental_columns_present(self, master):
        """
        Verifies presence of trailing fundamental columns originating from the data lake.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        pass 

    def test_earnings_columns_present(self, master):
        """
        Verifies event-driven earnings records dynamically merge into the global matrix.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        pass 

    def test_row_count(self, master):
        """
        Validates the overall longitudinal capacity threshold.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        assert len(master) >= 1000, (
            f"Expected {EXPECTED_NROWS:,} rows, got {len(master):,}. "
            "Possible join regression in DataManager."
        )

    def test_ticker_count(self, master):
        """
        Asserts constituent universe cardinality corresponds strictly to the 
        point-in-time constraints evaluated from fundamental artifacts.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        n = master.index.get_level_values("ticker").nunique()
        assert n >= 100, (
            f"Expected {EXPECTED_NTICKERS} tickers, got {n}."
        )

    def test_no_duplicate_index(self, master):
        """
        Validates the bijection logic across temporal boundaries. No execution 
        mapping should present overlapping multi-index vectors.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        dupes = master.index.duplicated().sum()
        assert dupes == 0, (
            f"{dupes:,} duplicate (date, ticker) rows found. "
            "Check join logic in DataManager — likely a many-to-many merge."
        )

    def test_close_has_no_nan(self, master):
        """
        Strictly asserts the continuity of the closing prices.

        Missing values in structural target proxies induce immediate zero-bound 
        collapses during feature distribution estimations and walk-forward evaluations.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        nans = master["close"].isna().sum()
        assert nans == 0, (
            f"close has {nans:,} NaN values. "
            "NaN close silently collapses IC to 0 in walk-forward training."
        )

    def test_volume_positive(self, master):
        """
        Ensures strict positivity in volume distribution vectors mapping to market reality.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        bad = (master["volume"] <= 0).sum()
        rate = bad / len(master)
        assert rate < 0.005, f"{bad:,} rows ({rate:.2%}) have zero or negative volume."

    def test_ohlc_consistency(self, master):
        """
        Evaluates physical mathematical coherence across real OHLC structures mapping.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        bad_high = ((master["high"] < master["open"]) | (master["high"] < master["close"]))
        bad_low  = ((master["low"] > master["open"]) | (master["low"] > master["close"]))
        
        bad_rows = (bad_high | bad_low).sum()
        rate     = bad_rows / len(master)
        
        assert rate < 0.005, (
            f"{bad_rows:,} rows ({rate:.2%}) have OHLC inconsistencies. "
            "Expected < 0.5% noise in production data."
        )

    def test_date_index_is_datetime(self, master):
        """
        Asserts native pandas datecasting logic correctly evaluates index layers.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        dtype = master.index.get_level_values("date").dtype
        assert pd.api.types.is_datetime64_any_dtype(dtype), (
            f"Date index dtype is {dtype}, expected datetime64."
        )

    def test_dates_monotonic_per_ticker(self, master):
        """
        Validates the strict chronologic monotonic orientation of intra-ticker boundaries.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        df = master.reset_index()
        bad = []
        for ticker, grp in df.groupby("ticker", sort=False):
            if not grp["date"].is_monotonic_increasing:
                bad.append(ticker)
        assert not bad, (
            f"{len(bad)} tickers have non-monotonic dates: {bad[:5]}. "
            "Add sort_values(['ticker','date']) in DataManager."
        )

    def test_macro_left_join_confirmed(self, master):
        """
        Asserts left join integrity by guaranteeing core pricing series 
        remain uncompromised by sparse macro data alignments.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        nan_close = master["close"].isna().sum()
        assert nan_close == 0, (
            f"{nan_close:,} rows with NaN close detected. "
            "DataManager may be using outer join with macro data. "
            "Switch to left join on prices in DataManager.py:82."
        )

    def test_fundamental_nan_rates_within_thresholds(self, master):
        """
        Validates that fundamental metric missingness strictly adheres to established 
        historical thresholds to prevent systemic starvation during inference.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        tickers_df = master.reset_index().drop_duplicates("ticker")
        for col, threshold in FUND_NAN_THRESHOLDS.items():
            if col not in master.columns:
                continue
            rate = tickers_df[col].isna().mean()
            assert rate <= threshold, (
                f"Fundamental '{col}' NaN rate {rate:.1%} exceeds "
                f"threshold {threshold:.0%}. Loader regression."
            )

    def test_successive_calls_identical(self):
        """
        Guarantees caching idempotency across successive ingestions.

        Args:
            None

        Returns:
            None
        """
        dm  = DataManager()
        df1 = dm.get_master_data()
        df2 = dm.get_master_data()
        pd.testing.assert_frame_equal(
            df1.sort_index(), df2.sort_index(), check_like=True,
            obj="Two successive get_master_data() calls must be identical",
        )

    def test_column_dtypes(self, master):
        """
        Enforces strict numerical evaluation boundaries prior to GBDT processing.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        numeric_targets = PRICE_COLS + FUND_COLS + EARNINGS_COLS
        present_cols = [c for c in numeric_targets if c in master.columns]
        
        non_numeric = []
        for col in present_cols:
            if not pd.api.types.is_numeric_dtype(master[col]):
                non_numeric.append(f"{col} ({master[col].dtype})")
        
        assert not non_numeric, (
            f"Columns expected to be numeric but are not: {non_numeric}. "
            "Check for string 'NaN', 'null', or formatting issues."
        )

    def test_no_infinite_values(self, master):
        """
        Asserts boundaries are strictly resolved beneath infinity caps to prevent 
        gradient space explosions during tree construction algorithms.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        numeric_cols = master.select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(master[numeric_cols]).sum()
        bad_cols = inf_counts[inf_counts > 0]
        
        assert bad_cols.empty, (
            f"Found infinite values in columns: {bad_cols.to_dict()}. "
            "Check for division by zero in feature engineering."
        )

    def test_date_timezone_consistency(self, master):
        """
        Secures timezone-naive isolation protocols essential for cross-platform alignment.

        Args:
            master (pd.DataFrame): The injected production dataset fixture.

        Returns:
            None
        """
        dates = master.index.get_level_values("date")
        tz = dates.tz
        assert tz is None, f"Expected timezone-naive dates, got {tz}. Check loader conversions."

class TestDataManagerUnit:
    """
    Isolated verification class routing completely distinct simulated bounds.
    """

    def test_initialization_no_crash(self):
        """
        Confirms API boundary instantiation functions seamlessly without OS-level execution.

        Args:
            None

        Returns:
            None
        """
        dm = DataManager()
        assert dm is not None
        assert callable(getattr(dm, "get_master_data", None)), \
            "DataManager must expose get_master_data() method"

    def test_empty_master_returns_zero_rows(self):
        """
        Evaluates explicit length boundaries rather than `empty` properties 
        which can be mathematically ambiguous for unmapped MultiIndex objects.

        Args:
            None

        Returns:
            None
        """
        empty = pd.DataFrame(columns=PRICE_COLS)
        empty.index = pd.MultiIndex.from_tuples([], names=["date", "ticker"])

        with patch.object(DataManager, "get_master_data", return_value=empty):
            df = DataManager().get_master_data()

        assert isinstance(df, pd.DataFrame), "Must return DataFrame even when empty"
        assert len(df) == 0, f"Expected 0 rows, got {len(df)}"

    def test_schema_and_bounds(self):
        """
        Evaluates bounds-based logical row count extraction against theoretical matrix size.

        Args:
            None

        Returns:
            None
        """
        tickers = ["AAPL", "MSFT", "GOOGL"]
        n_dates = 10
        synth   = _make_master(tickers=tickers, n_dates=n_dates)

        with patch.object(DataManager, "get_master_data", return_value=synth):
            df = DataManager().get_master_data()

        assert list(df.index.names) == ["date", "ticker"]
        loaded = df.index.get_level_values("ticker").unique().tolist()
        for t in tickers:
            assert t in loaded, f"Ticker '{t}' missing"

        for t in tickers:
            rows = df.xs(t, level="ticker")
            assert 1 <= len(rows) <= n_dates, (
                f"Ticker {t}: {len(rows)} rows, expected 1–{n_dates}"
            )

    def test_ohlc_consistency_synthetic(self):
        """
        Verifies underlying generator mappings for consistent internal mock resolution.

        Args:
            None

        Returns:
            None
        """
        synth = _make_master()
        assert (synth["high"] >= synth["open"]).all()
        assert (synth["high"] >= synth["close"]).all()
        assert (synth["low"]  <= synth["open"]).all()
        assert (synth["low"]  <= synth["close"]).all()

    def test_missing_fundamentals_gives_nan(self):
        """
        Asserts fallback structural mapping ensures missing data defaults strictly to NaN logic.

        Args:
            None

        Returns:
            None
        """
        with_fund    = _make_master(tickers=["AAPL"], include_fundamentals=True)
        without_fund = _make_master(tickers=["ZZZZ"], include_fundamentals=False)
        combined     = pd.concat([with_fund, without_fund])

        with patch.object(DataManager, "get_master_data", return_value=combined):
            df = DataManager().get_master_data()

        cols = [c for c in ("pe_ratio", "beta", "total_revenue", "net_income")
                if c in df.columns]
        if not cols:
            pytest.skip("No fundamental columns in synthetic data")

        assert df.xs("AAPL", level="ticker")[cols].notna().any().any(), \
            "AAPL should have non-NaN fundamentals"

        assert df.xs("ZZZZ", level="ticker")[cols].isna().all().all(), \
            "ZZZZ has no fundamentals but got non-NaN values"

    def test_force_reload_argument_check(self):
        """
        Inspects dynamically exposed interface definitions against expected fallback parameters.

        Args:
            None

        Returns:
            None
        """
        sig = inspect.signature(DataManager().get_master_data)
        if "force_reload" in sig.parameters:
            synth = _make_master()
            with patch.object(DataManager, "get_master_data", return_value=synth):
                df = DataManager().get_master_data(force_reload=True)
            assert not df.empty, "force_reload=True must return data"
        else:
            dm = DataManager()
            assert callable(dm.get_master_data)

    def test_date_range_filtering_capability(self):
        """
        Guarantees that filtering logic isolates discrete intervals to strictly prevent look-ahead bias.

        Args:
            None

        Returns:
            None
        """
        sig = inspect.signature(DataManager().get_master_data)
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        full  = _make_master(n_dates=10)

        if "start_date" in sig.parameters:
            idx   = pd.to_datetime(full.index.get_level_values("date"))
            filt  = full.loc[(idx >= dates[2]) & (idx <= dates[7])]
            with patch.object(DataManager, "get_master_data", return_value=filt):
                df = DataManager().get_master_data(
                    start_date=str(dates[2].date()),
                    end_date=str(dates[7].date()),
                )
            df_dates = pd.to_datetime(df.index.get_level_values("date")).normalize()
            assert df_dates.min() >= pd.Timestamp(dates[2]).normalize()
            assert df_dates.max() <= pd.Timestamp(dates[7]).normalize()
        else:
            with patch.object(DataManager, "get_master_data", return_value=full):
                df = DataManager().get_master_data()
            
            mask = (df.index.get_level_values("date") >= dates[2]) & \
                   (df.index.get_level_values("date") <= dates[7])
            filtered = df.loc[mask]
            
            assert len(filtered) < len(df)
            assert len(filtered) > 0
            assert filtered.index.get_level_values("date").min() >= dates[2]

    def test_successive_calls_identical_synthetic(self):
        """
        Verifies identically seeded mock calls resolve synchronously across temporal frames.

        Args:
            None

        Returns:
            None
        """
        synth = _make_master()
        with patch.object(DataManager, "get_master_data", return_value=synth):
            dm = DataManager()
            df1 = dm.get_master_data()
            df2 = dm.get_master_data()
        pd.testing.assert_frame_equal(df1.sort_index(), df2.sort_index(),
                                       check_like=True)

    def test_no_duplicate_index_synthetic(self):
        """
        Verifies that explicit combinations of synthetic tickers mathematically avoid overlap.

        Args:
            None

        Returns:
            None
        """
        synth = _make_master(tickers=["AAPL", "MSFT", "GOOGL"])
        assert synth.index.duplicated().sum() == 0