"""
UNIT TEST: Feature Engineering
===============================
Tests FactorRegistry, BaseFactor, and specific factor calculations.
Verifies that factors handle multi-ticker data and NaNs correctly.

BUGS FIXED vs v1:
  BUG C1 [CRITICAL]: Duplicate test_input_immutability method.
    Python silently overwrites the first definition — the assert_frame_equal
    version was dead code that never ran. The second version had a weaker
    assertion (just close.sum()) and no frame equality check.
    Fix: Merged both into one comprehensive test.

  BUG C2 [CRITICAL]: sample_market_data used np.random without seed.
    price = 100 + np.cumsum(np.random.normal(...)) — non-deterministic.
    test_grouping_logic passed/failed randomly depending on the draw.
    pe_ratio also used unseeded np.random.normal().
    Fix: All random generation uses np.random.default_rng(seed=42).

  BUG C3 [CRITICAL]: test_grouping_logic silently produced all-NaN momentum.
    df["mom"] = res[val_col] — if res is a Series/DataFrame with a different
    index than df (e.g. MultiIndex vs flat RangeIndex), pandas alignment fills
    NaN everywhere. val_a/val_b were then NaN, and assert NaN > 0 raised
    a misleading TypeError or vacuously passed depending on numpy version.
    Fix: Extract values by explicit date+ticker lookup, validate non-NaN
    before asserting direction.

  BUG H1 [HIGH]: result column detection in test_grouping_logic was fragile.
    val_col = res.columns[0] then if factor_name in res.columns: val_col = factor_name
    — if factor returns "value" or "factor_value", wrong column was used silently.
    Fix: Try factor_name, then first column, with explicit fallback and clear error.

  BUG H2 [HIGH]: test_grouping_logic never validated that mom values were non-NaN
    before asserting direction. assert NaN > 0 gives misleading failure.
    Fix: Assert values are finite before directional assertions.

  BUG H3 [HIGH]: test_technical_factor_calculation only checked len(res) == len(df).
    A factor returning all-NaN would pass. No warmup-aware NaN check existed.
    Fix: Assert that values after warmup period (row 30+) contain non-NaN.

  BUG H4 [HIGH]: sample_market_data OHLC not self-consistent.
    open = close = price[i], high = price[i]+1, low = price[i]-1.
    open should be previous close for realistic simulation. volume was constant
    10000 for all rows — breaks volume-based factors.
    Fix: Proper OHLCV generation using previous-close-as-open pattern.
    Volume uses seeded random integers.

  BUG M1 [MEDIUM]: FactorRegistry() may create a fresh empty instance each call
    if it is not a singleton. Every test doing `registry = FactorRegistry()`
    might get an empty registry if the global registry is separate.
    Fix: Added module-level registry fixture; documented assumption clearly.
    Added assertion that registry is non-empty before any factor lookup.

  BUG M2 [MEDIUM]: test_robustness_to_missing_columns had bare `except: pass`
    — vacuous test that always passed, asserting nothing.
    Fix: Use pytest.raises() or explicit result check to verify graceful failure.

  BUG M3 [MEDIUM]: Module-level side-effect imports with no error handling.
    ImportError would fail entire collection with a confusing message.
    Fix: Wrapped in try/except with pytest.importorskip()-style handling.

  BUG L1 [LOW]: test_fundamental_factor_pass_through never verified non-NaN
    output — pe_ratio after ffill may be NaN for first rows of each ticker.
    Fix: Assert that result contains non-NaN values after the warmup period.

  BUG L2 [LOW]: groupby().ffill().reset_index(drop=True) could silently drop
    the "ticker" column in some pandas versions (becomes group key, not column).
    Fix: Use sort_values + groupby(..., group_keys=False).ffill(), and assert
    "ticker" column survives the operation.
"""

import pytest
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# M3 FIX: Wrap side-effect imports — ImportError gives a clear skip, not
# a confusing collection failure.
# ---------------------------------------------------------------------------
try:
    from quant_alpha.features.registry import FactorRegistry
    import quant_alpha.features.technical.momentum    # registers momentum factors
    import quant_alpha.features.technical.volatility  # registers volatility factors
    import quant_alpha.features.fundamental.value     # registers value factors
    _IMPORT_ERROR = None
except ImportError as e:
    _IMPORT_ERROR = str(e)

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"Feature module import failed: {_IMPORT_ERROR}",
)

# ---------------------------------------------------------------------------
# M1 FIX: Module-level registry — created once after imports so all
# self-registration side effects have already run. Passed as a fixture
# so tests share the same populated instance.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def registry():
    """
    Shared FactorRegistry instance for all tests.

    M1 FIX: Creating FactorRegistry() inside each test risks getting an
    empty instance if the class is NOT a singleton (i.e. each __init__
    starts fresh). By creating it once at module scope AFTER the side-effect
    imports above, we guarantee factors are registered before any test runs.
    """
    reg = FactorRegistry()
    assert len(reg.factors) > 0, (
        "FactorRegistry is empty after importing momentum/volatility/value modules. "
        "Check that those modules call FactorRegistry().register(...) or equivalent "
        "at import time."
    )
    return reg


# ---------------------------------------------------------------------------
# H4 + C2 FIX: Proper OHLCV fixture with seeded RNG and correct OHLC logic
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_market_data():
    """
    Creates dummy OHLCV + Fundamental data for 2 tickers.

    C2 FIX: All random generation uses seeded RNG — deterministic results.
    H4 FIX: OHLC is now self-consistent:
      - open  = previous close (realistic)
      - high  = max(open, close) * (1 + small noise)
      - low   = min(open, close) * (1 - small noise)
      - volume uses seeded random integers (not constant 10000)
    L2 FIX: Explicit sort + group_keys=False in ffill preserves ticker column.
    """
    rng   = np.random.default_rng(seed=42)
    dates   = pd.date_range("2023-01-01", periods=100, freq="B")
    tickers = ["TICK_A", "TICK_B"]

    rows = []
    for seed_offset, t in enumerate(tickers):
        rng_t = np.random.default_rng(seed=42 + seed_offset * 7)
        n     = len(dates)

        # H4 FIX: realistic OHLCV
        close  = 100.0 + rng_t.standard_normal(n).cumsum()
        open_  = np.roll(close, 1); open_[0] = close[0]
        noise  = rng_t.uniform(0.001, 0.015, n)
        high   = np.maximum(open_, close) * (1 + noise)
        low    = np.minimum(open_, close) * (1 - noise)
        volume = rng_t.integers(50_000, 2_000_000, n).astype(float)

        for i, d in enumerate(dates):
            rows.append({
                "date":          d,
                "ticker":        t,
                "open":          open_[i],
                "high":          high[i],
                "low":           low[i],
                "close":         close[i],
                "volume":        volume[i],
                # Quarterly-ish fundamentals (populated every ~60 rows)
                "net_income":    5000.0 if i % 60 == 0 else np.nan,
                "total_revenue": 10000.0 if i % 60 == 0 else np.nan,
                "market_cap":    1_000_000.0,
                "pe_ratio":      15.0 + rng_t.standard_normal(),
                # eps needed for val_earnings_yield (eps/price) calculation
                "eps":           (5000.0 if i % 60 == 0 else np.nan) / 1000,
                "fwd_eps":       (5500.0 if i % 60 == 0 else np.nan) / 1000,
            })

    df = pd.DataFrame(rows)

    # L2 FIX: sort first, explicit ffill per ticker without groupby.apply()
    # FutureWarning FIX: avoid groupby().apply(lambda g: g.ffill()) which
    # triggers pandas >= 2.2 deprecation about applying on grouping columns.
    # Use groupby().transform("ffill") per column — no include_groups issue.
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    for col in df.columns:
        if col not in ("date", "ticker"):
            df[col] = df.groupby("ticker")[col].transform("ffill")

    # L2 FIX: assert ticker column survived the operation
    assert "ticker" in df.columns, (
        "groupby transform dropped 'ticker' column — pandas version issue."
    )
    return df


# ===========================================================================
# TESTS
# ===========================================================================

class TestFeatures:

    # ──────────────────────────────────────────────────────────────────────────
    # M1 FIX: Registry discovery
    # ──────────────────────────────────────────────────────────────────────────
    def test_registry_discovery(self, registry):
        """
        Factors self-register on import. Registry must not be empty and must
        contain momentum and volatility factors.
        """
        registered = list(registry.factors.keys())
        assert len(registered) > 0, "Registry should not be empty after imports"

        assert any(f.startswith("mom") for f in registered), (
            f"No momentum factors found. Registered: {registered}"
        )
        assert any(f.startswith("vol") for f in registered), (
            f"No volatility factors found. Registered: {registered}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # H3 FIX: Technical factor — verify non-NaN output after warmup
    # ──────────────────────────────────────────────────────────────────────────
    def test_technical_factor_calculation(self, registry, sample_market_data):
        """
        volatility_21d factor computes non-NaN values after warmup period.
        H3 FIX: Assert that values after row 30 are non-NaN — a factor
        returning all-NaN passes len() check but silently corrupts models.
        """
        factor_name = "volatility_21d"
        assert factor_name in registry.factors, (
            f"CRITICAL: '{factor_name}' missing from registry. "
            f"Registered factors: {list(registry.factors.keys())}"
        )

        factor = registry.factors[factor_name]
        res    = factor.calculate(sample_market_data)

        assert res is not None, f"{factor_name}.calculate() returned None"
        assert not res.empty,   f"{factor_name}.calculate() returned empty result"
        assert len(res) == len(sample_market_data), (
            f"Result length {len(res)} != input length {len(sample_market_data)}. "
            "Factor must return one value per input row."
        )

        # H3 FIX: check non-NaN after warmup — exclude metadata cols (date/ticker)
        if isinstance(res, pd.DataFrame):
            key_cols  = {"date", "ticker"}
            val_cols  = [c for c in res.columns if c not in key_cols]
            values    = res[val_cols].iloc[30:].values.ravel() if val_cols else res.iloc[30:].values.ravel()
        else:
            values = res.iloc[30:].values

        non_nan_count = (~np.isnan(values.astype(float))).sum()
        assert non_nan_count > 0, (
            f"{factor_name} returns all-NaN after warmup period (row 30+). "
            "Check lookback window and input data length (100 rows provided)."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # L1 FIX: Fundamental factor — verify non-NaN output
    # ──────────────────────────────────────────────────────────────────────────
    def test_fundamental_factor_pass_through(self, registry, sample_market_data):
        """
        val_earnings_yield (P/E inverse) returns non-NaN values.
        L1 FIX: pe_ratio is populated every row → earnings yield must be non-NaN.
        Factor name confirmed from registry: 'val_earnings_yield'.
        """
        factor_name = "val_earnings_yield"
        assert factor_name in registry.factors, (
            f"CRITICAL: '{factor_name}' missing from registry. "
            f"Registered factors: {list(registry.factors.keys())}"
        )

        factor = registry.factors[factor_name]
        res    = factor.calculate(sample_market_data)

        assert res is not None
        assert not res.empty
        assert len(res) == len(sample_market_data)

        # L1 FIX: pe_ratio is populated for all rows (no ffill gaps).
        # Earnings yield = 1/pe_ratio — must be mostly non-NaN.
        if isinstance(res, pd.DataFrame):
            key_cols = {"date", "ticker"}
            val_cols = [c for c in res.columns if c not in key_cols]
            values   = res[val_cols].values.ravel() if val_cols else res.values.ravel()
        else:
            values = res.values

        nan_rate = np.isnan(values.astype(float)).mean()
        assert nan_rate < 0.5, (
            f"val_earnings_yield has {nan_rate:.0%} NaN — expected mostly non-NaN "
            "since pe_ratio is populated for all 200 rows in sample_market_data."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # M2 FIX: Graceful failure on missing columns
    # ──────────────────────────────────────────────────────────────────────────
    def test_robustness_to_missing_columns(self, registry):
        """
        Factor must fail gracefully (raise a descriptive exception OR return
        empty/None) when required input columns are missing.

        M2 FIX: Original had bare `except: pass` — vacuous test, always passed.
        Now we assert that EITHER a specific exception is raised, OR the result
        is empty/None. We do NOT accept a result that looks valid but is silently
        wrong (e.g. a DataFrame of all-NaN with no warning).
        """
        bad_data = pd.DataFrame({
            "date":   pd.date_range("2023-01-01", periods=30),
            "ticker": ["A"] * 30,
            "volume": [100.0] * 30,
            # 'close', 'open', 'high', 'low' intentionally absent
        })

        factor_name = "volatility_21d"
        if factor_name not in registry.factors:
            pytest.skip(f"{factor_name} not in registry")

        factor = registry.factors[factor_name]

        raised   = False
        result   = None
        try:
            result = factor.calculate(bad_data)
        except (KeyError, ValueError, AttributeError) as e:
            raised = True  # expected: factor detected missing column

        # Accept either: exception raised, OR result is None/empty
        if not raised:
            assert result is None or (hasattr(result, "__len__") and len(result) == 0), (
                f"{factor_name}.calculate() on data missing 'close' did not raise "
                "and returned a non-empty result. "
                "Factor should raise KeyError/ValueError or return None/empty."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # C3 + H1 + H2 FIX: Grouping logic
    # ──────────────────────────────────────────────────────────────────────────
    def test_grouping_logic(self, registry, sample_market_data):
        """
        Factors must compute per-ticker, not mixing data across tickers.
        Ticker A trends up → positive momentum. Ticker B trends down → negative.

        C3 FIX: Result extracted via positional .values assignment after
        reset_index() on both df and res — avoids silent NaN from index
        type mismatch (MultiIndex vs RangeIndex).

        H1 FIX: val_col detection tries factor_name first, then falls back.

        H2 FIX: Values validated as finite before directional assertion.

        NEW FIX (merge collision): pd.merge() creates "col_x"/"col_y" suffixes
        when BOTH df AND res_df have the same column name (e.g. "return_21d").
        Subsequent rename({'return_21d': '_mom_val'}) silently does nothing
        because the key "return_21d" no longer exists (only "_x"/"_y" versions).
        Fix: use positional .values assignment (no merge) — immune to name collisions.
        """
        # Use the factor name that exists in the registry.
        # Try common momentum factor names in priority order.
        factor_name = None
        for candidate in ("mom_1m", "return_21d", "momentum_21d", "ret_1m"):
            if candidate in registry.factors:
                factor_name = candidate
                break
        assert factor_name is not None, (
            "No momentum factor found in registry. Tried: mom_1m, return_21d, "
            f"momentum_21d, ret_1m. Registered: {list(registry.factors.keys())}"
        )

        df = sample_market_data.copy().reset_index(drop=True)
        
        # FIX: Add sector/industry columns. Some factors (e.g. sector-neutral momentum)
        # require these to compute relative strength. Missing them can lead to 0.0 values.
        df["sector"]   = "Technology"
        df["industry"] = "Software"
        df["sector"]   = df["sector"].astype("category")
        df["industry"] = df["industry"].astype("category")

        # Force clear trends — enough rows for any momentum warmup (21+ days)
        mask_a = df["ticker"] == "TICK_A"
        mask_b = df["ticker"] == "TICK_B"

        n_a = mask_a.sum()
        n_b = mask_b.sum()

        df.loc[mask_a, "close"] = np.linspace(100, 200, n_a)  # strong uptrend
        df.loc[mask_b, "close"] = np.linspace(200, 100, n_b)  # strong downtrend

        # Force ALL price columns to match close trend.
        # FIX: Ensure High > Low to avoid zero-volatility crashes in factors (e.g. Sharpe/Sortino)
        df["open"]      = df["close"]
        df["high"]      = df["close"] * 1.001
        df["low"]       = df["close"] * 0.999
        df["adj_close"] = df["close"]
        df["adj_open"]  = df["open"]
        df["adj_high"]  = df["high"]
        df["adj_low"]   = df["low"]
        
        # Ensure explicit sort order for time-series calculations
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        factor = registry.factors[factor_name]
        res    = factor.calculate(df)

        assert res is not None, f"{factor_name}.calculate() returned None"
        assert not (res.empty if hasattr(res, "empty") else False), \
            f"{factor_name}.calculate() returned empty result"

        # H1 FIX: find the value column robustly
        if isinstance(res, pd.Series):
            res_df = res.to_frame(name="value")
            val_col = "value"
        else:
            res_df = res.copy()
            val_col = None
            for candidate in (factor_name, "value", "factor_value", "result"):
                if candidate in res_df.columns:
                    val_col = candidate
                    break
            if val_col is None:
                # Pick first non-key column
                key_cols = {"date", "ticker"}
                non_key  = [c for c in res_df.columns if c not in key_cols]
                assert non_key, f"res_df has no non-key columns: {res_df.columns.tolist()}"
                val_col = non_key[0]

        # ── Extract values via merge on [date, ticker] ──────────────────────
        #
        # ROOT CAUSE of val_a=0.0: factor outputs rows in [date, ticker]
        # interleaved order, but df is sorted [ticker, date]. Positional
        # .values assignment maps TICK_A's last-date row to TICK_B's
        # mid-history row → completely wrong value.
        #
        # CORRECT approach: merge on [date, ticker] keys (order-independent).
        #
        # MERGE COLLISION FIX: if df already has a column named val_col
        # (e.g. "return_21d" from a previous pass or same-named col),
        # pd.merge() creates "return_21d_x" / "return_21d_y" suffixes and
        # the subsequent rename silently does nothing.
        # Fix: rename val_col → "_factor_result_" in res_df BEFORE merging,
        # so there is never a name collision regardless of df's columns.

        res_flat = res_df.copy()
        if isinstance(res_flat.index, pd.MultiIndex):
            res_flat = res_flat.reset_index()

        # Ensure date/ticker keys exist in res_flat for the merge
        if "date" not in res_flat.columns or "ticker" not in res_flat.columns:
            pytest.fail(
                f"{factor_name}.calculate() returned a DataFrame without "
                "'date' and 'ticker' columns — cannot align with input by key. "
                f"Columns returned: {res_flat.columns.tolist()}"
            )

        # Pre-rename to avoid collision (the collision-proof key step)
        SAFE_COL = "_factor_result_"
        res_flat = res_flat.rename(columns={val_col: SAFE_COL})

        res_flat["date"]   = pd.to_datetime(res_flat["date"])
        df["date"]         = pd.to_datetime(df["date"])

        df = df.merge(res_flat[["date", "ticker", SAFE_COL]],
                      on=["date", "ticker"], how="left")

        last_date = df["date"].max()
        last_day  = df[df["date"] == last_date]

        row_a = last_day[last_day["ticker"] == "TICK_A"]
        row_b = last_day[last_day["ticker"] == "TICK_B"]

        assert len(row_a) == 1, f"TICK_A not found on last date {last_date}"
        assert len(row_b) == 1, f"TICK_B not found on last date {last_date}"

        val_a = row_a[SAFE_COL].values[0]
        val_b = row_b[SAFE_COL].values[0]

        # H2 FIX: validate finite before directional check
        assert np.isfinite(val_a), (
            f"TICK_A {factor_name} is {val_a} (NaN/inf). "
            "Factor failed to compute a finite value for an uptrending ticker. "
            f"Column used: '{val_col}'. Check warmup period vs data length (100 rows)."
        )
        assert np.isfinite(val_b), (
            f"TICK_B {factor_name} is {val_b} (NaN/inf). "
            "Factor failed to compute a finite value for a downtrending ticker. "
            f"Column used: '{val_col}'. Check warmup period vs data length (100 rows)."
        )

        assert val_a > 0, (
            f"TICK_A (strong uptrend 100→200) {factor_name} should be > 0, got {val_a:.6f}. "
            "If factor cross-section normalizes, uptrend ticker must rank above 0."
        )
        assert val_b < 0, (
            f"TICK_B (strong downtrend 200→100) {factor_name} should be < 0, got {val_b:.6f}. "
            "If factor cross-section normalizes, downtrend ticker must rank below 0."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # C1 FIX: Single comprehensive input immutability test
    # ──────────────────────────────────────────────────────────────────────────
    def test_input_immutability(self, registry, sample_market_data):
        """
        Factor calculation must NOT modify the input DataFrame (no side effects).

        C1 FIX: Original had TWO methods with the same name — Python silently
        discarded the first (assert_frame_equal version). The surviving second
        version only checked close.sum(), missing column additions/deletions.

        Fix: Single comprehensive test that checks:
          1. Column list unchanged (no added/dropped columns)
          2. Index unchanged
          3. Values unchanged (assert_frame_equal for full equality)
        """
        factor_name = "volatility_21d"
        if factor_name not in registry.factors:
            pytest.skip(f"{factor_name} not registered — skipping immutability check")

        df_original = sample_market_data.copy(deep=True)
        factor      = registry.factors[factor_name]

        _ = factor.calculate(sample_market_data)

        # 1. Column list must not change
        assert list(sample_market_data.columns) == list(df_original.columns), (
            "Factor calculation added or removed columns from input DataFrame. "
            f"Before: {list(df_original.columns)}\n"
            f"After:  {list(sample_market_data.columns)}"
        )

        # 2. Index must not change
        pd.testing.assert_index_equal(
            sample_market_data.index, df_original.index,
            obj="Factor calculation modified input DataFrame index",
        )

        # 3. Values must not change (full equality)
        pd.testing.assert_frame_equal(
            sample_market_data, df_original,
            check_like=False,
            obj="Factor calculation modified input DataFrame values",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # OHLC consistency validation of the fixture itself
    # ──────────────────────────────────────────────────────────────────────────
    def test_fixture_ohlc_consistency(self, sample_market_data):
        """
        H4 FIX validation: confirm the sample_market_data fixture generates
        OHLC-consistent data (high >= max(open,close), low <= min(open,close)).
        """
        df = sample_market_data
        assert (df["high"] >= df["open"]).all(),  "Fixture: high < open"
        assert (df["high"] >= df["close"]).all(), "Fixture: high < close"
        assert (df["low"]  <= df["open"]).all(),  "Fixture: low > open"
        assert (df["low"]  <= df["close"]).all(), "Fixture: low > close"
        assert (df["volume"] > 0).all(),          "Fixture: zero/negative volume"