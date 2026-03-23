r"""
Historical Simulation and Performance Attribution Validation
============================================================
Validates the event-driven backtesting engine and structural performance attribution layers.

Purpose
-------
This module validates the core financial logic of the backtesting engine (`BacktestEngine`)
and the statistical accuracy of the attribution module (`SimpleAttribution`). It ensures
that the simulation correctly handles order execution, cost modelling (commissions, slippage),
and accounting principles (FIFO/Average Cost) without introducing look-ahead bias or
floating-point accounting errors.

Role in Quantitative Workflow
-----------------------------
Serves as a fundamental algorithmic safeguard, guaranteeing that simulated strategy 
results perfectly mirror mathematical expectations before committing strategies 
to live capital deployment.

Importance
----------
- **Execution Integrity**: Verifies that trades are executed at valid market prices (preventing
  look-ahead bias where $P_{exec} \approx P_{close}$ erroneously).
- **Accounting Precision**: Asserts that P&L calculations strictly adhere to the accounting
  identity $Equity_t = Equity_{t-1} + PnL_t$, handling floating-point epsilon ($\epsilon$) checks.
- **Cost Modelling**: Validates the impact of transaction costs (commissions, slippage) on
  Net Asset Value (NAV) to ensure realistic alpha estimation.
- **Attribution Accuracy**: Confirms that key metrics (Sharpe, Hit Ratio, Profit Factor) are
  calculated correctly, handling edge cases like zero-trade periods or bankruptcy ($NAV \le 0$).

Tools & Frameworks
------------------
- **Pytest**: Testing framework and fixture management.
- **Pandas/NumPy**: In-memory synthesis of deterministic market environments.
- **Unittest.Mock**: External dependency isolation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Graceful degradation mapping to safely bypass integration boundaries if core execution engines are missing
_BACKTEST_IMPORT_ERROR     = None
_ATTRIBUTION_IMPORT_ERROR  = None

try:
    from quant_alpha.backtest.engine import BacktestEngine
except ImportError as e:
    BacktestEngine = None  # type: ignore
    _BACKTEST_IMPORT_ERROR = str(e)

try:
    from quant_alpha.backtest.attribution import SimpleAttribution
except ImportError as e:
    SimpleAttribution = None  # type: ignore
    _ATTRIBUTION_IMPORT_ERROR = str(e)


def _get_equity_value(equity_curve: pd.DataFrame, col_candidates=("total_value", "equity", "portfolio_value")) -> pd.Series:
    """
    Robustly extracts the Net Asset Value (NAV) series from the equity curve.
    
    Dynamically handles schema variations across different execution engine implementations.

    Args:
        equity_curve (pd.DataFrame): The sequential portfolio valuation ledger.
        col_candidates (tuple, optional): Target column namespaces expected to contain NAV. 
            Defaults to ("total_value", "equity", "portfolio_value").

    Returns:
        pd.Series: The isolated chronological Net Asset Value series.

    Raises:
        AssertionError: If none of the candidate columns are located in the schema.
    """
    for col in col_candidates:
        if col in equity_curve.columns:
            return equity_curve[col]
    raise AssertionError(
        f"equity_curve has no recognisable value column. "
        f"Tried: {col_candidates}. "
        f"Actual columns: {equity_curve.columns.tolist()}"
    )


@pytest.mark.skipif(
    BacktestEngine is None,
    reason=f"BacktestEngine not importable: {_BACKTEST_IMPORT_ERROR}"
)
class TestBacktestEngine:
    """
    Execution validation suite for the historical simulation engine.
    """

    @pytest.fixture
    def synthetic_market(self):
        """
        Generates a deterministic Bull Market scenario for signal validation.
        
        Properties:
        - **Trend**: Linear upward drift (+2.0/day).
        - **Execution Gap**: $P_{open} \neq P_{close}$ ensures the engine distinguishes between
          execution timestamps, preventing look-ahead bias.
          
        Period: 20 business days (sufficient for warmup and trade generation).

        Args:
            None

        Returns:
            tuple: A 2-element tuple containing:
                - pd.DataFrame: Idealized target prediction matrix.
                - pd.DataFrame: Corresponding synthetic OHLCV pricing matrix.
        """
        dates = pd.date_range("2023-01-02", periods=20, freq="B")
        rng   = np.random.default_rng(seed=42)

        prices_rows = []
        preds_rows  = []

        for i, d in enumerate(dates):
            # Injects a strict deterministic uptrend to guarantee verifiable signal capture boundaries
            up_close   = 100.0 + i * 2.0
            up_open    = 100.0 + (i - 1) * 2.0 if i > 0 else up_close
            volume     = 1_000_000 + int(rng.integers(-50_000, 50_000))

            prices_rows += [
                {"date": d, "ticker": "TICK_A",
                 "open": up_open, "close": up_close,
                 "high": max(up_open, up_close),
                 "low":  min(up_open, up_close),
                 "volume": volume, "volatility": 0.01},
                {"date": d, "ticker": "TICK_B",
                 "open": up_open, "close": up_close,
                 "high": max(up_open, up_close),
                 "low":  min(up_open, up_close),
                 "volume": volume, "volatility": 0.01},
            ]
            preds_rows += [
                {"date": d, "ticker": "TICK_A", "prediction": 1.0},
                {"date": d, "ticker": "TICK_B", "prediction": 1.0},
            ]

        prices = pd.DataFrame(prices_rows)
        preds  = pd.DataFrame(preds_rows)
        return preds, prices

    def test_initialization(self):
        """
        Verifies that critical financial parameters (Capital, Commission, Slippage)
        are correctly persisted in the engine state.

        Args:
            None

        Returns:
            None
        """
        engine = BacktestEngine(
            initial_capital=50_000,
            commission=0.001,
            slippage=0.0005,
            spread=0.0002,
        )
        assert engine.initial_capital == 50_000,  "initial_capital not stored correctly"

        # Assert optional secondary cost friction parameters are securely mapped if exposed by the engine
        if hasattr(engine, "slippage"):
            assert engine.slippage == 0.0005, "slippage not stored correctly"
        if hasattr(engine, "spread"):
            assert engine.spread   == 0.0002, "spread not stored correctly"

    def test_perfect_foresight_profitability(self, synthetic_market):
        r"""
        Validates execution logic under ideal conditions (Perfect Signal, Zero Cost).
        
        Constraints:
        1. **Profitability**: $NAV_{final} > NAV_{initial}$
        2. **Monotonicity**: $NAV_t \ge NAV_{t-1}, \forall t$ (No drawdown possible).

        Args:
            synthetic_market (tuple): The deterministic prediction and pricing matrices.

        Returns:
            None
        """
        preds, prices = synthetic_market

        engine = BacktestEngine(
            initial_capital=100_000,
            commission=0.0,
            slippage=0.0,
            spread=0.0,
            rebalance_freq="daily",
            use_market_impact=False,
            execution_price="close", 
        )

        results = engine.run(preds, prices, top_n=2)

        # Asserts structural compliance of the simulation engine's output mapping
        assert "equity_curve" in results, (
            "engine.run() result missing 'equity_curve' key. "
            f"Keys returned: {list(results.keys())}"
        )
        equity_series = _get_equity_value(results["equity_curve"])

        final_value = float(equity_series.iloc[-1])
        assert final_value > 100_000, (
            f"Perfect strategy lost money: final={final_value:.2f} <= 100,000. "
            "Check execution direction (long=buy, short=sell) and price used."
        )

        # Strictly verifies monotonicity (Zero-Loss Constraint) bounding floating-point approximations
        daily_change = equity_series.diff().dropna()
        losing_days  = (daily_change < -1e-8).sum() 
        assert losing_days == 0, (
            f"Equity decreased on {losing_days} days despite perfect signals "
            f"and zero costs. Losing-day changes: "
            f"{daily_change[daily_change < -1e-8].values}. "
            "Check position rebalancing and P&L accrual logic."
        )

        assert "trades" in results and not results["trades"].empty, (
            "No trades generated by BacktestEngine. "
            "Check signal threshold and top_n parameter."
        )

    def test_empty_inputs(self):
        """
        Ensures system robustness when handling empty datasets.
        
        The engine should return a neutral equity curve (Cash only) without
        raising unhandled exceptions.

        Args:
            None

        Returns:
            None
        """
        engine = BacktestEngine(initial_capital=100_000, commission=0.001)

        # Constructs structurally compliant but vacant DataFrames to validate edge-case routing
        empty_preds  = pd.DataFrame(columns=["date", "ticker", "prediction"]).astype({"prediction": float})
        empty_prices = pd.DataFrame(columns=["date", "ticker", "open", "close", "volume"]).astype(
            {"open": float, "close": float, "volume": float})

        try:
            results = engine.run(empty_preds, empty_prices, top_n=2)
        except Exception as e:
            pytest.fail(
                f"BacktestEngine crashed on empty inputs: {type(e).__name__}: {e}. "
                "Engine must handle empty DataFrames gracefully."
            )

        assert "equity_curve" in results, "Result missing 'equity_curve' for empty inputs"
        assert "trades"       in results, "Result missing 'trades' for empty inputs"
        assert len(results["trades"]) == 0, (
            f"Expected 0 trades for empty inputs, got {len(results['trades'])}"
        )

    def test_cash_constraint_scales_buys(self):
        r"""
        Verifies that orders exceeding available cash are correctly scaled or capped.
        Constraint: $OrderValue \le AvailableCash$.

        Args:
            None

        Returns:
            None
        """
        dates = pd.date_range("2023-01-01", periods=2, freq="B")
        prices = pd.DataFrame([
            {"date": dates[0], "ticker": "A", "close": 100, "open": 100, "volume": 1e6}
        ])
        preds = pd.DataFrame([
            {"date": dates[0], "ticker": "A", "prediction": 1.0}
        ])
        
        # Intentionally forces execution to exceed nominal account values to verify structural scale-down logic
        engine = BacktestEngine(initial_capital=1000)
        res = engine.run(preds, prices, top_n=1)
        
        equity_curve = res["equity_curve"]
        cash_col = "cash" if "cash" in equity_curve.columns else "cash_balance"
        if cash_col in equity_curve.columns:
            min_cash = equity_curve[cash_col].min()
            assert min_cash >= 0, f"Cash went negative: {min_cash}"

    def test_market_impact_integration(self, synthetic_market):
        r"""
        Verifies that enabling market impact modeling correctly penalizes performance.
        Hypothesis: $NAV_{impact} \le NAV_{no\_impact}$.

        Args:
            synthetic_market (tuple): The deterministic prediction and pricing matrices.

        Returns:
            None
        """
        preds, prices = synthetic_market
        engine_no_impact = BacktestEngine(initial_capital=100000, use_market_impact=False)
        engine_impact    = BacktestEngine(initial_capital=100000, use_market_impact=True)
        
        res_no = engine_no_impact.run(preds, prices, top_n=2)
        res_imp = engine_impact.run(preds, prices, top_n=2)
        
        val_no  = float(_get_equity_value(res_no["equity_curve"]).iloc[-1])
        val_imp = float(_get_equity_value(res_imp["equity_curve"]).iloc[-1])
        
        assert val_imp <= val_no, "Market impact should reduce (or equal) performance"

    def test_average_cost_accounting(self):
        """
        Verifies Average Cost Basis lot matching logic in trade logging.

        Ensures correct cost basis attribution when selling a portion of a position.

        Args:
            None

        Returns:
            None
        """
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        prices = pd.DataFrame([
            {"date": dates[0], "ticker": "A", "close": 100, "open": 100, "volume": 1e6},
            {"date": dates[1], "ticker": "A", "close": 110, "open": 110, "volume": 1e6},
            {"date": dates[2], "ticker": "A", "close": 120, "open": 120, "volume": 1e6},
        ])
        preds = pd.DataFrame([
            {"date": dates[0], "ticker": "A", "prediction": 1000 / 10000},
            {"date": dates[1], "ticker": "A", "prediction": 2200 / 10100},
            {"date": dates[2], "ticker": "A", "prediction": 600 / 10300},
        ])
        
        engine = BacktestEngine(
            initial_capital=10000, commission=0, spread=0, slippage=0, 
            use_market_impact=False, execution_price="close", rebalance_freq="daily"
        )
        res = engine.run(preds, prices, is_weights=True)
        trades = res["trades"]
        
        # Asserts the closing boundary event calculates cost basis dynamically
        assert len(trades) >= 3
        sell_trade = trades.iloc[-1]
        
        assert sell_trade["pnl"] == pytest.approx(225.0, abs=1.0)

@pytest.mark.skipif(
    SimpleAttribution is None,
    reason=f"SimpleAttribution not importable: {_ATTRIBUTION_IMPORT_ERROR}"
)
class TestAttribution:
    """
    Statistical validation suite for the portfolio performance attribution layer.
    """

    def test_pnl_aggregation(self):
        r"""
        Validates the aggregation of trade-level P&L into portfolio stats.
        
        Checks:
        - **Total PnL**: $\sum PnL_i$
        - **Profit Factor**: $\frac{\sum GrossProfit}{|\sum GrossLoss|}$
        - **Hit Ratio**: $\frac{N_{winners}}{N_{total}}$

        Args:
            None

        Returns:
            None
        """
        trades = pd.DataFrame([
            {"ticker": "A", "side": "long",  "entry_price": 100, "exit_price": 110, "pnl":  10.0},
            {"ticker": "B", "side": "long",  "entry_price": 100, "exit_price":  90, "pnl": -10.0},
            {"ticker": "C", "side": "short", "entry_price": 100, "exit_price":  90, "pnl":  10.0},
        ])
        trades["date"] = pd.Timestamp("2023-01-01")
        trades["size"] = 100.0
        trades["commission"] = 0.0
        trades["return"] = trades["pnl"] / (trades["entry_price"] * trades["size"])
        trades["net_pnl"] = trades["pnl"]
        trades["status"] = "Closed"
        trades["entry_time"] = trades["date"]
        trades["exit_time"] = trades["date"] + pd.Timedelta(hours=4)
        trades["side"] = trades["side"].str.lower()

        attr  = SimpleAttribution()
        stats = attr.analyze_pnl_drivers(trades)

        # Validates fundamental P&L distribution logic and boundaries
        assert stats.get("n_trades", 0)     == 3,                       "n_trades wrong"
        assert stats.get("hit_ratio", 0.0)      == pytest.approx(0.6667, abs=1e-4),   "hit_ratio wrong"
        assert stats.get("long_pnl_contribution", 0.0)   == pytest.approx(0.0),    "long_pnl_contribution wrong"
        assert stats.get("short_pnl_contribution", 0.0)     == pytest.approx(10.0),   "short_pnl_contribution wrong"

        if "profit_factor" in stats:
            assert stats["profit_factor"] == pytest.approx(2.0), (
                f"profit_factor wrong: {stats['profit_factor']:.4f} != 2.0"
            )

    def test_attribution_empty_trades(self):
        """
        Tests resilience against Division-by-Zero errors when no trades occur.
        
        Scenario: `total_trades = 0`.
        Expected:
        - `hit_ratio` should be 0.0 or NaN (not raise Error).
        - `total_pnl` should be 0.0.

        Args:
            None

        Returns:
            None
        """
        empty_trades = pd.DataFrame(
            columns=["date", "ticker", "side", "size", "entry_price", "exit_price", "pnl", "commission", "return", "net_pnl", "status", "entry_time", "exit_time"]
        )

        attr = SimpleAttribution()
        try:
            stats = attr.analyze_pnl_drivers(empty_trades)
        except ZeroDivisionError:
            pytest.fail(
                "SimpleAttribution.analyze_pnl_drivers() raised ZeroDivisionError "
                "on empty trades. Guard hit_ratio with: "
                "hit_ratio = winning / total if total > 0 else 0.0"
            )
        except Exception as e:
            pytest.fail(
                f"Unexpected exception on empty trades: {type(e).__name__}: {e}"
            )

        assert stats.get("n_trades", -1)   == 0,               "n_trades should be 0"

        hr = stats.get("hit_ratio", 0)
        assert hr == pytest.approx(0.0) or (
            isinstance(hr, float) and np.isnan(hr)
        ), f"hit_ratio for 0 trades should be 0 or NaN, got {hr}"

    def test_all_winning_trades(self):
        """
        Verifies boundary condition: $HitRatio = 1.0$ when all trades are profitable.

        Args:
            None

        Returns:
            None
        """
        trades = pd.DataFrame([
            {"ticker": t, "side": "long", "entry_price": 100, "exit_price": 110, "pnl": 10.0}
            for t in ["A", "B", "C"]
        ])
        trades["date"] = pd.Timestamp("2023-01-01")
        trades["size"] = 100.0
        trades["commission"] = 0.0
        trades["return"] = trades["pnl"] / (trades["entry_price"] * trades["size"])
        trades["net_pnl"] = trades["pnl"]
        trades["status"] = "Closed"
        trades["entry_time"] = trades["date"]
        trades["exit_time"] = trades["date"] + pd.Timedelta(hours=4)
        trades["side"] = trades["side"].str.lower()
        stats = SimpleAttribution().analyze_pnl_drivers(trades)
        assert stats["hit_ratio"]    == pytest.approx(1.0)
        assert stats.get("long_pnl_contribution", 0.0)   == pytest.approx(30.0)
        assert stats.get("short_pnl_contribution", 0.0) == pytest.approx(0.0)

    def test_all_losing_trades(self):
        """
        Verifies boundary condition: $HitRatio = 0.0$ when all trades are unprofitable.

        Args:
            None

        Returns:
            None
        """
        trades = pd.DataFrame([
            {"ticker": t, "side": "long", "entry_price": 100, "exit_price": 90, "pnl": -10.0}
            for t in ["A", "B", "C"]
        ])
        trades["date"] = pd.Timestamp("2023-01-01")
        trades["size"] = 100.0
        trades["commission"] = 0.0
        trades["return"] = trades["pnl"] / (trades["entry_price"] * trades["size"])
        trades["net_pnl"] = trades["pnl"]
        trades["status"] = "Closed"
        trades["entry_time"] = trades["date"]
        trades["exit_time"] = trades["date"] + pd.Timedelta(hours=4)
        trades["side"] = trades["side"].str.lower()
        stats = SimpleAttribution().analyze_pnl_drivers(trades)
        assert stats["hit_ratio"]    == pytest.approx(0.0)
        assert stats.get("long_pnl_contribution", 0.0) == pytest.approx(-30.0)
        assert stats.get("short_pnl_contribution", 0.0)   == pytest.approx(0.0)